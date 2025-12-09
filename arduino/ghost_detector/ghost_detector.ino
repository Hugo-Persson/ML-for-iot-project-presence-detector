// Presence detector on Arduino Nano 33 BLE using MicroTFLite
// Pipeline matches notebook: 0.5s audio chunks -> FFT binning -> 2s rolling window (4 chunks) -> 1-output sigmoid

#include <Arduino.h>
#include <PDM.h>
#include <arm_math.h>
#include <MicroTFLite.h>

#include "net.h" // TFLite model array (seizure_model)

// Audio / feature params (match training)
constexpr int SAMPLE_RATE = 16000;
constexpr float CHUNK_DURATION_S = 0.5f;
constexpr int CHUNK_SAMPLES = static_cast<int>(SAMPLE_RATE * CHUNK_DURATION_S); // 8000
constexpr int SEGMENT_CHUNKS = 4;                                               // 2s window
constexpr int CHUNK_FEATURES = 207;                                             // binned FFT features per chunk
constexpr int MODEL_INPUTS = SEGMENT_CHUNKS * CHUNK_FEATURES;                   // 828
constexpr int MODEL_OUTPUTS = 1;

// FFT setup (RFFT needs power-of-two length; we zero-pad to 8192)
constexpr int FFT_LEN = 8192;
constexpr int FFT_COMPLEX_LEN = FFT_LEN;      // arm_rfft_fast_f32 uses N complex-length buffer
constexpr int FFT_MAG_BINS = FFT_LEN / 2 + 1; // number of real FFT bins

// Tensor arena for MicroTFLite
constexpr int kTensorArenaSize = 40 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// PDM buffers
constexpr int PDM_READ_SAMPLES = 512; // samples per PDM.read (tune as needed)
int16_t pdm_buffer[PDM_READ_SAMPLES];
int16_t audio_buffer[CHUNK_SAMPLES]; // accumulates 0.5s chunk
volatile size_t audio_index = 0;
volatile bool chunk_ready = false;

// Feature storage for rolling window
float chunk_features[CHUNK_FEATURES];
float segment_window[SEGMENT_CHUNKS][CHUNK_FEATURES];
int stored_chunks = 0; // how many chunks currently in window (max 4)

// FFT workspace
arm_rfft_fast_instance_f32 fft_instance;
float fft_input[FFT_LEN];
float fft_output[FFT_COMPLEX_LEN];
float fft_magnitude[FFT_MAG_BINS];

// LED for presence indicator (prefer red channel if defined)
#ifdef LEDR
constexpr int PRESENCE_LED_PIN = LEDR;
#else
constexpr int PRESENCE_LED_PIN = LED_BUILTIN;
#endif
constexpr float PRESENCE_THRESHOLD = 0.5f; // sigmoid threshold

void onPDMdata();
void compute_chunk_features(const int16_t *pcm, float *out_features);
void push_chunk_and_maybe_infer(const float *features);
void run_inference(const float window[SEGMENT_CHUNKS][CHUNK_FEATURES]);

void setup()
{
    pinMode(PRESENCE_LED_PIN, OUTPUT);
    digitalWrite(PRESENCE_LED_PIN, LOW);

    Serial.begin(115200);
    while (!Serial)
        ;
    Serial.println("Presence detector starting...");

    // Init FFT (zero-padded to 8192)
    arm_rfft_fast_init_f32(&fft_instance, FFT_LEN);

    // Init TensorFlow Lite Micro
    if (!ModelInit(seizure_model, tensor_arena, kTensorArenaSize))
    {
        Serial.println("Model initialization failed!");
        while (true)
            delay(1000);
    }
    ModelPrintInputTensorDimensions();
    ModelPrintOutputTensorDimensions();

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
    // When a 0.5s audio chunk is ready, compute features and run inference
    if (chunk_ready)
    {
        chunk_ready = false; // clear early to allow next fill
        compute_chunk_features(audio_buffer, chunk_features);
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
            audio_buffer[audio_index++] = pdm_buffer[i];
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
    // Copy PCM to float and zero-pad to FFT_LEN
    for (int i = 0; i < CHUNK_SAMPLES; ++i)
    {
        fft_input[i] = static_cast<float>(pcm[i]);
    }
    for (int i = CHUNK_SAMPLES; i < FFT_LEN; ++i)
    {
        fft_input[i] = 0.0f;
    }

    // Run RFFT
    arm_rfft_fast_f32(&fft_instance, fft_input, fft_output, 0);

    // Magnitude for first FFT_MAG_BINS bins
    // Bin 0 (DC)
    fft_magnitude[0] = fabsf(fft_output[0]);
    // Bins 1..N-2 (complex)
    for (int k = 1; k < FFT_MAG_BINS - 1; ++k)
    {
        float re = fft_output[2 * k];
        float im = fft_output[2 * k + 1];
        fft_magnitude[k] = sqrtf(re * re + im * im);
    }
    // Nyquist bin
    fft_magnitude[FFT_MAG_BINS - 1] = fabsf(fft_output[1]);

    // Binning parameters
    const float bin_width = static_cast<float>(SAMPLE_RATE) / static_cast<float>(FFT_LEN);
    auto hz_to_bin = [&](float hz) -> int
    {
        int idx = static_cast<int>(floorf(hz / bin_width));
        if (idx < 0)
            return 0;
        if (idx > FFT_MAG_BINS)
            return FFT_MAG_BINS;
        return idx;
    };

    const int bin_320 = min(hz_to_bin(320.0f), FFT_MAG_BINS);
    const int bin_3200 = min(hz_to_bin(3200.0f), FFT_MAG_BINS);

    auto merge_region = [&](int start, int end, float target_bw_hz, float *dst, int &dst_index)
    {
        const int len = max(0, end - start);
        if (len == 0)
            return;
        const int bins_per = max(1, static_cast<int>(roundf(target_bw_hz / bin_width)));
        const int usable = (len / bins_per) * bins_per;
        for (int i = 0; i < usable; i += bins_per)
        {
            float acc = 0.0f;
            for (int j = 0; j < bins_per; ++j)
            {
                acc += fft_magnitude[start + i + j];
            }
            dst[dst_index++] = acc / static_cast<float>(bins_per);
        }
    };

    int idx = 0;
    merge_region(0, bin_320, 4.0f, out_features, idx);               // low (~4 Hz bins)
    merge_region(bin_320, bin_3200, 32.0f, out_features, idx);       // mid (~32 Hz bins)
    merge_region(bin_3200, FFT_MAG_BINS, 128.0f, out_features, idx); // high (~128 Hz bins)

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

// Flatten window and run TFLM inference; drive LED on presence
void run_inference(const float window[SEGMENT_CHUNKS][CHUNK_FEATURES])
{
    // Flatten to match model input (1 x 4 x 207 x 1 => 828 floats)
    int flat_idx = 0;
    for (int r = 0; r < SEGMENT_CHUNKS; ++r)
    {
        for (int c = 0; c < CHUNK_FEATURES; ++c)
        {
            if (!ModelSetInput(window[r][c], flat_idx))
            {
                Serial.print("Failed to set input at index ");
                Serial.println(flat_idx);
                return;
            }
            flat_idx++;
        }
    }

    if (!ModelRunInference())
    {
        Serial.println("RunInference failed!");
        return;
    }

    float presence_prob = ModelGetOutput(0);
    bool presence = presence_prob >= PRESENCE_THRESHOLD;
    digitalWrite(PRESENCE_LED_PIN, presence ? HIGH : LOW);

    Serial.print("Presence prob: ");
    Serial.println(presence_prob, 4);
    Serial.print("Presence: ");
    Serial.println(presence ? "YES" : "NO");
}
