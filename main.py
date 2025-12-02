import struct
import csv
import wave
import serial
import argparse
import time
import subprocess

# Packet types
PACKET_TYPE_IMU = 0x01
PACKET_TYPE_AUDIO = 0x02

# Packet sizes (after sync + type byte)
IMU_PACKET_SIZE = 4 + 24  # timestamp + 6 floats
AUDIO_HEADER_SIZE = 4 + 2  # timestamp + sample count

AUDIO_RATE_HZ = 16000

ser = serial.Serial("/dev/cu.usbmodem21401", 115200)


def find_sync_and_type(ser: serial.Serial) -> int:
    """Scan until we find sync bytes, then return the packet type."""
    while True:
        b = ser.read(1)
        if b == b"\xaa":
            b2 = ser.read(1)
            if b2 == b"\x55":
                packet_type = ser.read(1)
                return packet_type[0]


def parse_imu_packet(data: bytes) -> tuple[int, list[float]] | None:
    """Parse IMU packet into (timestamp, imu_data)."""
    if len(data) != IMU_PACKET_SIZE:
        return None

    timestamp = struct.unpack("<I", data[0:4])[0]
    imu_data = list(struct.unpack("<6f", data[4:28]))
    return timestamp, imu_data


def parse_audio_header(data: bytes) -> tuple[int, int] | None:
    """Parse audio header into (timestamp, sample_count)."""
    if len(data) != AUDIO_HEADER_SIZE:
        return None

    timestamp = struct.unpack("<I", data[0:4])[0]
    sample_count = struct.unpack("<H", data[4:6])[0]
    return timestamp, sample_count


def parse_audio_samples(data: bytes, sample_count: int) -> list[int] | None:
    """Parse audio samples from raw bytes."""
    expected_size = sample_count * 2
    if len(data) != expected_size:
        return None

    return list(struct.unpack(f"<{sample_count}h", data))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record IMU and audio data from serial device"
    )
    parser.add_argument("name", help="Name/identifier for the recording session")
    parser.add_argument(
        "label",
        choices=["presence", "no_presence"],
        help="Presence label for the recording",
    )
    parser.add_argument(
        "recording_time",
        type=float,
        help="Recording duration in seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_filename = f"{args.name}_{args.label}.csv"
    wav_filename = f"{args.name}_{args.label}.wav"

    csv_file = open(csv_filename, "w", newline="")
    writer = csv.writer(csv_file)

    # WAV file: 16 kHz, mono, 16-bit
    wav_file = wave.open(wav_filename, "wb")
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)  # 2 bytes = 16-bit
    wav_file.setframerate(AUDIO_RATE_HZ)

    # CSV header (IMU only)
    csv_header = ["t_ms", "ax", "ay", "az", "gx", "gy", "gz"]
    writer.writerow(csv_header)

    print(
        f"Recording '{args.name}' with label '{args.label}' for {args.recording_time}s"
    )
    print(f"Waiting for data on {ser.port}...")

    start_time = time.time()

    try:
        while time.time() - start_time < args.recording_time:
            packet_type = find_sync_and_type(ser)

            if packet_type == PACKET_TYPE_IMU:
                data = ser.read(IMU_PACKET_SIZE)
                result = parse_imu_packet(data)
                if result is None:
                    continue

                timestamp, imu_data = result
                row = [timestamp] + imu_data
                writer.writerow(row)
                csv_file.flush()

            elif packet_type == PACKET_TYPE_AUDIO:
                header_data = ser.read(AUDIO_HEADER_SIZE)
                header = parse_audio_header(header_data)
                if header is None:
                    continue

                timestamp, sample_count = header
                if sample_count == 0 or sample_count > 512:
                    continue

                audio_data_bytes = ser.read(sample_count * 2)
                audio_data = parse_audio_samples(audio_data_bytes, sample_count)
                if audio_data is None:
                    continue

                audio_bytes = struct.pack(f"<{sample_count}h", *audio_data)
                wav_file.writeframes(audio_bytes)

        print("\nRecording complete.")
    except KeyboardInterrupt:
        print("\nStopping early...")
    finally:
        csv_file.close()
        wav_file.close()
        print(f"Saved: {csv_filename}, {wav_filename}")
        subprocess.run(
            ["afplay", "/System/Library/Sounds/Submarine.aiff"],
            check=False,
        )


if __name__ == "__main__":
    main()
