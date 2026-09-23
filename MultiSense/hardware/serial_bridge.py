#!/usr/bin/env python3
"""
MultiSense: High-Speed Arduino Uno Serial-to-API Bridge
Reads real-time sensor streams from Arduino Uno USB Serial:
  IR: 0
  ACC: 0, 0, 0
  Light: -2.00 lux
Sends them to the MultiSense ML API, stores them in MongoDB Atlas,
and transmits clinical alerts back to the Arduino Uno (Buzzer/LED).
"""

import sys
import time
import argparse
import urllib.request
import urllib.error
import glob

# Try importing pyserial
try:
    import serial
except ImportError:
    print("[!] pyserial not found. Installing pyserial...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyserial"])
    import serial

DEFAULT_API_URL = "http://127.0.0.1:8000/api/arduino/stream?format=text"
CLOUD_API_URL = "https://multisense-ai-pslv.onrender.com/api/arduino/stream?format=text"


def find_arduino_port():
    """Auto-detects USB serial port connected to Arduino Uno."""
    patterns = [
        "/dev/cu.usbmodem*",
        "/dev/tty.usbmodem*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
        "COM[0-9]*"
    ]
    for pattern in patterns:
        ports = glob.glob(pattern)
        if ports:
            return ports[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="MultiSense Arduino Serial Bridge")
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g. /dev/cu.usbmodem1101 or COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument("--api", type=str, default=DEFAULT_API_URL, help="API endpoint URL")
    parser.add_argument("--cloud", action="store_true", help="Use cloud Render API URL instead of localhost")
    args = parser.parse_args()

    api_url = CLOUD_API_URL if args.cloud else args.api

    port = args.port or find_arduino_port()
    if not port:
        print("[!] No Arduino Uno port detected automatically.")
        print("    Please specify your port manually, e.g.:")
        print("    python3 serial_bridge.py --port /dev/cu.usbmodem1101")
        sys.exit(1)

    print("=" * 70)
    print("  MULTISENSE ARDUINO UNO SERIAL-TO-API & MONGODB BRIDGE")
    print("=" * 70)
    print(f"  - Arduino Port:  {port}")
    print(f"  - Baud Rate:     {args.baud}")
    print(f"  - Target API:    {api_url}")
    print("=" * 70)

    try:
        ser = serial.Serial(port, args.baud, timeout=2)
        time.sleep(2) # Wait for Arduino Uno reset
        ser.reset_input_buffer()
        print("[+] Serial connection established. Listening for sensor stream...\n")
    except Exception as e:
        print(f"[!] Failed to open serial port {port}: {e}")
        sys.exit(1)

    buffer_lines = []

    try:
        while True:
            if ser.in_waiting > 0:
                try:
                    raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue

                if not raw_line:
                    continue

                buffer_lines.append(raw_line)

                # Check if we have collected the full 3-sensor reading
                if len(buffer_lines) >= 3 or ("Light:" in raw_line and len(buffer_lines) >= 2):
                    payload_text = "\n".join(buffer_lines)
                    buffer_lines = [] # Reset buffer

                    # Send to MultiSense ML API & MongoDB
                    req = urllib.request.Request(
                        api_url,
                        data=payload_text.encode("utf-8"),
                        headers={"Content-Type": "text/plain"}
                    )

                    try:
                        with urllib.request.urlopen(req, timeout=5) as response:
                            resp_text = response.read().decode("utf-8").strip()

                            # Parse response: e.g. PRED:ArmFlapping|CONF:0.45|RISK:MEDIUM|BUZZER:0|DB:SAVED
                            parts = dict(item.split(":", 1) for item in resp_text.split("|") if ":" in item)
                            pred = parts.get("PRED", "Unknown")
                            conf = parts.get("CONF", "0.0")
                            risk = parts.get("RISK", "LOW")
                            buzzer = parts.get("BUZZER", "0")
                            db_stat = parts.get("DB", "SAVED")

                            # Formatted terminal output
                            timestamp = time.strftime("%H:%M:%S")
                            risk_color = "\033[91m" if risk == "HIGH" else ("\033[93m" if risk == "MEDIUM" else "\033[92m")
                            reset = "\033[0m"

                            print(f"[{timestamp}] Sensor Stream Ingested:")
                            for l in payload_text.split("\n"):
                                print(f"    {l}")
                            print(f"    ==> ML: \033[1m{pred}\033[0m (Conf: {conf}) | Alert: {risk_color}[{risk}]\033[0m | DB: \033[96m{db_stat}\033[0m")

                            # Send buzzer alert signal back to Arduino
                            if buzzer == "1":
                                ser.write(b"BUZZER:1\n")
                            else:
                                ser.write(b"BUZZER:0\n")

                    except urllib.error.URLError as ue:
                        print(f"[!] API Network error: {ue}")
                    except Exception as e:
                        print(f"[!] Error forwarding reading: {e}")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[+] Exiting Serial Bridge. Closing port...")
        ser.close()


if __name__ == "__main__":
    main()
