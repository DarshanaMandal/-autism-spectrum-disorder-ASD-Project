/*
 * MultiSense ASD - Arduino Uno Multi-Sensor Telemetry
 * Sensors Supported:
 *   1. MAX30102 / Pulse Sensor (IR reading)
 *   2. MPU6050 3-Axis Accelerometer (ax, ay, az)
 *   3. BH1750 / LDR Ambient Light Sensor (lux)
 *
 * Output Format sent via Serial:
 *   IR: 0
 *   ACC: 0, 0, 0
 *   Light: -2.00 lux
 *
 * Incoming Alert:
 *   Listens for "BUZZER:1" or "ALERT:HIGH" to activate Pin 8 (Buzzer) and Pin 13 (LED).
 */

#include <Wire.h>

// Pin Configuration
const int BUZZER_PIN = 8;
const int LED_PIN = 13;
const int IR_ANALOG_PIN = A0;      // Optional analog IR fallback if MAX30102 not on I2C
const int LDR_ANALOG_PIN = A1;     // Optional LDR fallback if BH1750 not on I2C

// MPU6050 Default I2C Address
const int MPU_ADDR = 0x68;

// Sensor Reading Variables
long irValue = 0;
float ax = 0.0, ay = 0.0, az = 0.0;
float lightLux = -2.00;

unsigned long lastSendTime = 0;
const unsigned long SEND_INTERVAL = 1000; // Send telemetry every 1 second

void setup() {
  Serial.begin(115200); // High speed serial
  while (!Serial && millis() < 3000); // Wait for serial on USB

  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  Wire.begin();

  // Initialize MPU6050 (Wake up from sleep mode)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0x00); // Wake up
  byte mpuStatus = Wire.endTransmission();

  // Self-test LED flash
  digitalWrite(LED_PIN, HIGH);
  delay(200);
  digitalWrite(LED_PIN, LOW);
}

void loop() {
  // 1. Read Sensors
  readSensors();

  // 2. Transmit Telemetry every interval
  if (millis() - lastSendTime >= SEND_INTERVAL) {
    sendTelemetry();
    lastSendTime = millis();
  }

  // 3. Check for Incoming Alerts from ML API Bridge
  checkIncomingAlerts();
}

void readSensors() {
  // --- MPU6050 Accelerometer Reading ---
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); // Starting at register 0x3B (ACCEL_XOUT_H)
  if (Wire.endTransmission(false) == 0) {
    Wire.requestFrom(MPU_ADDR, 6, true);
    if (Wire.available() >= 6) {
      int16_t raw_ax = (Wire.read() << 8) | Wire.read();
      int16_t raw_ay = (Wire.read() << 8) | Wire.read();
      int16_t raw_az = (Wire.read() << 8) | Wire.read();

      // Convert raw LSB to g (for default +/- 2g range: divide by 16384.0)
      ax = raw_ax / 16384.0;
      ay = raw_ay / 16384.0;
      az = raw_az / 16384.0;
    }
  } else {
    // If MPU not connected, zero reading
    ax = 0.0;
    ay = 0.0;
    az = 0.0;
  }

  // --- IR Sensor Reading ---
  // If using analog photodiode/IR on A0:
  int irRaw = analogRead(IR_ANALOG_PIN);
  // If no finger detected or sensor open, keep 0
  if (irRaw < 30) {
    irValue = 0;
  } else {
    irValue = irRaw;
  }

  // --- Light Sensor Reading ---
  // If BH1750 not responding, use uncalibrated offset reading (-2.00 lux) or LDR
  int ldrRaw = analogRead(LDR_ANALOG_PIN);
  if (ldrRaw < 10) {
    lightLux = -2.00; // Sensor dark / negative offset reading
  } else {
    lightLux = map(ldrRaw, 10, 1023, 0, 800);
  }
}

void sendTelemetry() {
  // Transmits the exact 3-line format recognized by MultiSense API:
  Serial.print("IR: ");
  Serial.println(irValue);

  Serial.print("ACC: ");
  Serial.print(ax, 2);
  Serial.print(", ");
  Serial.print(ay, 2);
  Serial.print(", ");
  Serial.println(az, 2);

  Serial.print("Light: ");
  Serial.print(lightLux, 2);
  Serial.println(" lux");
}

void checkIncomingAlerts() {
  if (Serial.available() > 0) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();

    // Check if ML predicted HIGH risk or BUZZER:1
    if (msg.indexOf("BUZZER:1") >= 0 || msg.indexOf("ALERT:HIGH") >= 0) {
      // Sound alarm and turn on LED
      digitalWrite(LED_PIN, HIGH);
      tone(BUZZER_PIN, 2000, 300); // 2kHz tone for 300ms
    } else if (msg.indexOf("BUZZER:0") >= 0 || msg.indexOf("ALERT:LOW") >= 0) {
      digitalWrite(LED_PIN, LOW);
      noTone(BUZZER_PIN);
    }
  }
}
