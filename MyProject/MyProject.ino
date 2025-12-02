#include <Arduino.h>
#include <PDM.h>
#include <Arduino_LSM9DS1.h>

// default number of output channels
static const char channels = 1;

// default PCM output frequency
static const int frequency = 16000;
// IMU data
float ax, ay, az, gx, gy, gz;
// Buffer to read samples into, each sample is 16-bits
short sampleBuffer[512];
// Binary packet markers
const uint8_t SYNC_BYTE_1 = 0xAA;
const uint8_t SYNC_BYTE_2 = 0x55;
const uint8_t PACKET_TYPE_IMU = 0x01;
const uint8_t PACKET_TYPE_AUDIO = 0x02;
// Number of audio samples read
volatile int samplesRead;
void sendImuPacket(uint32_t timestamp) {
  Serial.write(SYNC_BYTE_1);
  Serial.write(SYNC_BYTE_2);
  Serial.write(PACKET_TYPE_IMU);

  // Timestamp (4 bytes, little-endian)
  Serial.write((uint8_t*)&timestamp, 4);

  // IMU data (6 floats = 24 bytes)
  Serial.write((uint8_t*)&ax, 4);
  Serial.write((uint8_t*)&ay, 4);
  Serial.write((uint8_t*)&az, 4);
  Serial.write((uint8_t*)&gx, 4);
  Serial.write((uint8_t*)&gy, 4);
  Serial.write((uint8_t*)&gz, 4);
}

void sendAudioPacket(uint32_t timestamp) {
  Serial.write(SYNC_BYTE_1);
  Serial.write(SYNC_BYTE_2);
  Serial.write(PACKET_TYPE_AUDIO);

  // Timestamp (4 bytes, little-endian)
  Serial.write((uint8_t*)&timestamp, 4);

  // Sample count (2 bytes, little-endian)
  uint16_t sampleCount = samplesRead;
  Serial.write((uint8_t*)&sampleCount, 2);

  // Audio samples (sampleCount × 2 bytes)
  Serial.write((uint8_t*)sampleBuffer, samplesRead * 2);
}


void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Configure the data receive callback
  PDM.onReceive(onPDMdata);

  // Optionally set the gain
  // Defaults to 20 on the BLE Sense and 24 on the Portenta Vision Shield
  PDM.setGain(80);

  // Initialize PDM with:
  // - one channel (mono mode)
  // - a 16 kHz sample rate for the Arduino Nano 33 BLE Sense
  // - a 32 kHz or 64 kHz sample rate for the Arduino Portenta Vision Shield
  if (!PDM.begin(channels, frequency)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }

  if (!IMU.begin()) {
    Serial.println("Failed to start IMU!");
    while (1);
  }
}

void loop() {
  // Wait for samples to be read
  if (samplesRead) {
    sendAudioPacket(millis());
    // Clear the read count
    samplesRead = 0;
  }

  // Read and send IMU data when available
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
    sendImuPacket(millis());
  }
}

/**
 * Callback function to process the data from the PDM microphone.
 * NOTE: This callback is executed as part of an ISR.
 * Therefore using `Serial` to print messages inside this function isn't supported.
 * */
void onPDMdata() {
  // Query the number of available bytes
  int bytesAvailable = PDM.available();

  // Read into the sample buffer
  PDM.read(sampleBuffer, bytesAvailable);

  // 16-bit, 2 bytes per sample
  samplesRead = bytesAvailable / 2;
}



