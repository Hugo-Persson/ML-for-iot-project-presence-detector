// Presence detector on Arduino Nano 33 BLE using MicroTFLite
// Pipeline matches notebook: 0.5s audio chunks -> FFT binning -> 2s rolling window (4 chunks) -> 1-output sigmoid

#include <Arduino.h>
#include <PDM.h>
#include <arduinoFFT.h>
#include <math.h>
#include <string.h>

// On-device training (online learning) using a tiny logistic regression.
// Train by sending labels over Serial (see setup()).

// Audio / feature params (match training)
constexpr int SAMPLE_RATE = 16000;
constexpr float CHUNK_DURATION_S = 0.5f;
constexpr int CHUNK_SAMPLES = static_cast<int>(SAMPLE_RATE * CHUNK_DURATION_S); // 8000
constexpr int SEGMENT_CHUNKS = 4;                                               // 2s window
constexpr int CHUNK_FEATURES = 207;                                             // binned FFT features per chunk
constexpr int MODEL_INPUTS = SEGMENT_CHUNKS * CHUNK_FEATURES;                   // 828
constexpr int MODEL_OUTPUTS = 1;

// FFT setup (needs power-of-two length; we zero-pad to 8192)
constexpr int FFT_LEN = 8192;
constexpr int FFT_MAG_BINS = FFT_LEN / 2 + 1; // number of real FFT bins


// PDM buffers
constexpr int PDM_READ_SAMPLES = 512; // samples per PDM.read (tune as needed)
int16_t pdm_buffer[PDM_READ_SAMPLES];
int16_t audio_capture[CHUNK_SAMPLES]; // accumulates 0.5s chunk (ISR writes here)
int16_t audio_process[CHUNK_SAMPLES]; // main loop copies here before processing
volatile size_t audio_index = 0;
volatile bool chunk_ready = false;

// Feature storage for rolling window
float chunk_features[CHUNK_FEATURES];
float segment_window[SEGMENT_CHUNKS][CHUNK_FEATURES];
int stored_chunks = 0; // how many chunks currently in window (max 4)

// FFT workspace (arduinoFFT uses separate real/imaginary arrays)
float fft_real[FFT_LEN];
float fft_imag[FFT_LEN];
float fft_magnitude[FFT_MAG_BINS];

// ArduinoFFT instance
ArduinoFFT<float> FFT = ArduinoFFT<float>(fft_real, fft_imag, FFT_LEN, SAMPLE_RATE);

// LED for presence indicator (prefer red channel if defined)
//#ifdef LEDR
//constexpr int PRESENCE_LED_PIN = LEDR;
//#else
constexpr int PRESENCE_LED_PIN = LED_BUILTIN;
//#endif
// Output post-processing: smoothing + hysteresis to reduce flicker/sensitivity.
constexpr float PROB_EMA_ALPHA = 0.25f;          // 0..1, higher = less smoothing
constexpr float PRESENCE_THRESHOLD_ON = 0.60f;   // turn on when EMA >= this
constexpr float PRESENCE_THRESHOLD_OFF = 0.40f;  // turn off when EMA <= this

float presence_prob_ema = 0.0f;
bool presence_state = false;
bool ema_initialized = false;

void onPDMdata();
void compute_chunk_features(const int16_t *pcm, float *out_features);
void push_chunk_and_maybe_infer(const float *features);
void run_inference(const float window[SEGMENT_CHUNKS][CHUNK_FEATURES]);

// Online logistic regression (828 weights + bias). This trains on-device.
// Uses pretrained weights from lr_weights.h
#include "lr_weights.h"

constexpr float LR_LEARNING_RATE = 0.05f;
constexpr float LR_L2 = 1e-4f;
constexpr float LR_EPS = 1e-6f;
constexpr int LR_STATUS_EVERY = 4; // print every N updates (2s window, so 4 -> ~2s)

float lr_weights[MODEL_INPUTS] = {0.0f};
float lr_bias = 0.0f;
int8_t lr_train_label = -1; // -1 = not training, 0/1 = label
uint32_t lr_updates = 0;

static float sigmoidf_stable(float x)
{
    if (x >= 0.0f)
    {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

static void lr_reset()
{
    static_assert(LR_MODEL_INPUTS == MODEL_INPUTS, "lr_weights.h has unexpected input size.");
    memcpy(lr_weights, LR_WEIGHTS_INIT, sizeof(lr_weights));
    lr_bias = LR_BIAS_INIT;
    lr_train_label = -1;
    lr_updates = 0;
    ema_initialized = false;
    presence_state = false;
}

static void handle_serial_training_commands()
{
    while (Serial.available() > 0)
    {
        const char c = static_cast<char>(Serial.read());
        if (c == '0')
        {
            digitalWrite(LEDR, LOW);
            digitalWrite(LEDB, HIGH);
            digitalWrite(LEDG, HIGH);
            lr_train_label = 0;
            Serial.println("Training label set to 0 (no_presence).");
        }
        else if (c == '1')
        {
            digitalWrite(LEDR, HIGH);
            digitalWrite(LEDB, HIGH);
            digitalWrite(LEDG, LOW);
            lr_train_label = 1;
            Serial.println("Training label set to 1 (presence).");
        }
        else if (c == 'x' || c == 'X')
        {
            digitalWrite(LEDR, HIGH);
            digitalWrite(LEDB, LOW);
            digitalWrite(LEDG, HIGH);
            lr_train_label = -1;
            Serial.println("Training paused (inference only).");
        }
        else if (c == 'r' || c == 'R')
        {
            digitalWrite(LEDR, LOW);
            digitalWrite(LEDB, LOW);
            digitalWrite(LEDG, LOW);
            delay(500);
            digitalWrite(LEDR, HIGH);
            digitalWrite(LEDB, LOW);
            digitalWrite(LEDG, HIGH);
            lr_reset();
            Serial.println("LogReg reset (reloaded pretrained init weights).");
        }
        else if (c == 'h' || c == 'H' || c == '?')
        {
            Serial.println("On-device training controls:");
            Serial.println("  '1' -> train as presence");
            Serial.println("  '0' -> train as no_presence");
            Serial.println("  'x' -> stop training");
            Serial.println("  'r' -> reset weights");
            Serial.println("  'h' -> help");
        }
    }
}

static float lr_predict_and_maybe_update(const float window[SEGMENT_CHUNKS][CHUNK_FEATURES])
{
    // Simple gain normalization to reduce sensitivity to overall loudness:
    // scale features by 1/mean(feature).
    float sum = 0.0f;
    for (int r = 0; r < SEGMENT_CHUNKS; ++r)
    {
        for (int c = 0; c < CHUNK_FEATURES; ++c)
            sum += window[r][c];
    }
    const float mean = sum / static_cast<float>(MODEL_INPUTS);
    const float inv = 1.0f / (mean + LR_EPS);

    float z = lr_bias;
    int flat_idx = 0;
    for (int r = 0; r < SEGMENT_CHUNKS; ++r)
    {
        for (int c = 0; c < CHUNK_FEATURES; ++c)
        {
            const float x = window[r][c] * inv;
            z += lr_weights[flat_idx++] * x;
        }
    }

    // Clamp logits for numerical stability.
    if (z > 20.0f)
        z = 20.0f;
    else if (z < -20.0f)
        z = -20.0f;

    const float p = sigmoidf_stable(z);

    if (lr_train_label == 0 || lr_train_label == 1)
    {
        const float y = static_cast<float>(lr_train_label);
        const float grad = (p - y); // dL/dz for BCE with sigmoid

        // SGD update
        flat_idx = 0;
        for (int r = 0; r < SEGMENT_CHUNKS; ++r)
        {
            for (int c = 0; c < CHUNK_FEATURES; ++c)
            {
                const float x = window[r][c] * inv;
                lr_weights[flat_idx] -= LR_LEARNING_RATE * (grad * x + LR_L2 * lr_weights[flat_idx]);
                flat_idx++;
            }
        }
        lr_bias -= LR_LEARNING_RATE * grad;
        lr_updates++;

        if ((lr_updates % LR_STATUS_EVERY) == 0)
        {
            const float p_clip = min(max(p, LR_EPS), 1.0f - LR_EPS);
            const float loss = -(y * logf(p_clip) + (1.0f - y) * logf(1.0f - p_clip));
            Serial.print("Train y=");
            Serial.print(lr_train_label);
            Serial.print(" p=");
            Serial.print(p, 4);
            Serial.print(" loss=");
            Serial.print(loss, 4);
            Serial.print(" updates=");
            Serial.println(lr_updates);
        }
    }

    return p;
}

void setup()
{
    pinMode(PRESENCE_LED_PIN, OUTPUT);
    pinMode(LEDB, OUTPUT);
    pinMode(LEDG, OUTPUT);
    digitalWrite(PRESENCE_LED_PIN, LOW);
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDB, LOW);
    digitalWrite(LEDG, HIGH);

    Serial.begin(115200);
    while (!Serial)
        ;
    Serial.println("Presence detector starting...");

    Serial.println("On-device training enabled (logistic regression).");
    Serial.println("Send '1' (presence) or '0' (no_presence) to train, 'x' to stop, 'r' to reset, 'h' for help.");
    lr_reset();
    Serial.println("Pretrained init: loaded from lr_weights.h");

    // Configure PDM microphone
    PDM.onReceive(onPDMdata);
    PDM.setGain(80);
    if (!PDM.begin(1, SAMPLE_RATE))
    {
        Serial.println("Failed to start PDM!");
        while (true)
            delay(1000);
    }
    Serial.println("PDM started.");
}

void loop()
{
    handle_serial_training_commands();

    // When a 0.5s audio chunk is ready, compute features and run inference
    if (chunk_ready)
    {
        // Copy the completed chunk to a separate buffer before clearing the flag.
        // This avoids the ISR overwriting samples while we compute features.
        noInterrupts();
        memcpy(audio_process, audio_capture, sizeof(audio_capture));
        chunk_ready = false; // allow ISR to start filling the next chunk
        interrupts();

        compute_chunk_features(audio_process, chunk_features);
        push_chunk_and_maybe_infer(chunk_features);
    }
}

// ISR: read PDM samples into rolling audio buffer
void onPDMdata()
{
    if (chunk_ready)
    {
        // Drop incoming audio until main loop consumes
        int bytes = PDM.available();
        if (bytes > 0)
        {
            bytes = min(bytes, (int)(PDM_READ_SAMPLES * sizeof(int16_t)));
            PDM.read(pdm_buffer, bytes);
        }
        return;
    }

    int bytesAvailable = PDM.available();
    if (bytesAvailable <= 0)
        return;
    int bytesToRead = min(bytesAvailable, (int)(PDM_READ_SAMPLES * sizeof(int16_t)));
    PDM.read(pdm_buffer, bytesToRead);
    int samples = bytesToRead / 2;

    for (int i = 0; i < samples; ++i)
    {
        if (audio_index < CHUNK_SAMPLES)
        {
            audio_capture[audio_index++] = pdm_buffer[i];
            if (audio_index >= CHUNK_SAMPLES)
            {
                audio_index = 0;
                chunk_ready = true;
                break;
            }
        }
        else
        {
            // Should not happen; reset
            audio_index = 0;
            break;
        }
    }
}

// Compute FFT magnitude features and bin them into 207-length vector
void compute_chunk_features(const int16_t *pcm, float *out_features)
{
    // Notebook uses float32 audio in [-1, 1] (soundfile normalizes 16-bit PCM).
    // Match that by scaling int16 samples to [-1, 1] before the FFT.
    constexpr float PCM_SCALE = 1.0f / 32768.0f;

    // Training used an 8000-sample FFT (0.5s @ 16kHz). We use a 8192-point FFT
    // with zero-padding; below we bin magnitudes to match the notebook's 207
    // features by mapping "training bins" to the closest 8192 FFT bin.
    constexpr int TRAIN_FFT_LEN = CHUNK_SAMPLES;               // 8000
    constexpr int TRAIN_FFT_BINS = TRAIN_FFT_LEN / 2 + 1;      // 4001
    constexpr int TRAIN_BIN_320 = 160;                         // 320 Hz / 2 Hz
    constexpr int TRAIN_BIN_3200 = 1600;                       // 3200 Hz / 2 Hz
    constexpr int TRAIN_BINS_PER_LOW = 2;                      // 4 Hz / 2 Hz
    constexpr int TRAIN_BINS_PER_MID = 16;                     // 32 Hz / 2 Hz
    constexpr int TRAIN_BINS_PER_HIGH = 64;                    // 128 Hz / 2 Hz

    auto train_bin_to_fft_bin = [&](int train_bin) -> int
    {
        // Map training bin (0..4000) at ~2 Hz spacing to FFT_LEN bin index.
        // Using rounding keeps the endpoints aligned (e.g., 4000 -> 4096).
        const int mapped = (train_bin * FFT_LEN + TRAIN_FFT_LEN / 2) / TRAIN_FFT_LEN;
        if (mapped < 0)
            return 0;
        if (mapped >= FFT_MAG_BINS)
            return FFT_MAG_BINS - 1;
        return mapped;
    };

    // Copy PCM to real array and zero-pad; clear imaginary array
    for (int i = 0; i < CHUNK_SAMPLES; ++i)
    {
        fft_real[i] = static_cast<float>(pcm[i]) * PCM_SCALE;
        fft_imag[i] = 0.0f;
    }
    for (int i = CHUNK_SAMPLES; i < FFT_LEN; ++i)
    {
        fft_real[i] = 0.0f;
        fft_imag[i] = 0.0f;
    }

    // Run FFT (in-place, results stored in fft_real/fft_imag)
    FFT.compute(FFTDirection::Forward);

    // Compute magnitudes for first FFT_MAG_BINS bins
    for (int k = 0; k < FFT_MAG_BINS; ++k)
    {
        fft_magnitude[k] = sqrtf(fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k]);
    }

    auto merge_region_train = [&](int start_train_bin, int end_train_bin, int bins_per, float *dst, int &dst_index)
    {
        const int len = max(0, end_train_bin - start_train_bin);
        if (len == 0)
            return;
        const int usable = (len / bins_per) * bins_per; // drop remainder to match notebook behavior
        for (int i = 0; i < usable; i += bins_per)
        {
            if (dst_index >= CHUNK_FEATURES)
                return;
            float acc = 0.0f;
            for (int j = 0; j < bins_per; ++j)
            {
                const int train_bin = start_train_bin + i + j;
                acc += fft_magnitude[train_bin_to_fft_bin(train_bin)];
            }
            dst[dst_index++] = acc / static_cast<float>(bins_per);
        }
    };

    int idx = 0;
    merge_region_train(0, TRAIN_BIN_320, TRAIN_BINS_PER_LOW, out_features, idx);
    merge_region_train(TRAIN_BIN_320, TRAIN_BIN_3200, TRAIN_BINS_PER_MID, out_features, idx);
    merge_region_train(TRAIN_BIN_3200, TRAIN_FFT_BINS, TRAIN_BINS_PER_HIGH, out_features, idx);

    // Safety: zero any remaining slots
    for (; idx < CHUNK_FEATURES; ++idx)
    {
        out_features[idx] = 0.0f;
    }
}

// Maintain rolling 4-chunk window and trigger inference
void push_chunk_and_maybe_infer(const float *features)
{
    if (stored_chunks < SEGMENT_CHUNKS)
    {
        memcpy(segment_window[stored_chunks], features, sizeof(float) * CHUNK_FEATURES);
        stored_chunks++;
    }
    else
    {
        // shift left by one chunk
        for (int i = 1; i < SEGMENT_CHUNKS; ++i)
        {
            memcpy(segment_window[i - 1], segment_window[i], sizeof(float) * CHUNK_FEATURES);
        }
        memcpy(segment_window[SEGMENT_CHUNKS - 1], features, sizeof(float) * CHUNK_FEATURES);
    }

    if (stored_chunks >= SEGMENT_CHUNKS)
    {
        run_inference(segment_window);
    }
}

// Run logistic regression inference; drive LED on presence
void run_inference(const float window[SEGMENT_CHUNKS][CHUNK_FEATURES])
{
    const float presence_prob = lr_predict_and_maybe_update(window);

    if (!ema_initialized)
    {
        presence_prob_ema = presence_prob;
        ema_initialized = true;
    }
    else
    {
        presence_prob_ema =
            PROB_EMA_ALPHA * presence_prob + (1.0f - PROB_EMA_ALPHA) * presence_prob_ema;
    }

    if (!presence_state && presence_prob_ema >= PRESENCE_THRESHOLD_ON)
        presence_state = true;
    else if (presence_state && presence_prob_ema <= PRESENCE_THRESHOLD_OFF)
        presence_state = false;

    digitalWrite(PRESENCE_LED_PIN, presence_state ? HIGH : LOW);

    Serial.print("Presence prob: ");
    Serial.println(presence_prob, 4);
    Serial.print("Presence EMA: ");
    Serial.println(presence_prob_ema, 4);
    Serial.print("Presence: ");
    Serial.println(presence_state ? "YES" : "NO");
}
