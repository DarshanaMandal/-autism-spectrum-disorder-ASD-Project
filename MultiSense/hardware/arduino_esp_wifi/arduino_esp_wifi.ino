/*
 * MultiSense ASD - Arduino Uno with ESP8266 / ESP32 WiFi HTTP Client
 * Sends live sensor readings directly to MultiSense API via HTTP POST/GET
 * and logs to MongoDB Atlas without requiring a PC bridge!
 */

#if defined(ESP8266)
  #include <ESP8266WiFi.h>
  #include <ESP8266HTTPClient.h>
  #include <WiFiClient.h>
#elif defined(ESP32)
  #include <WiFi.h>
  #include <HTTPClient.h>
#else
  #error "Please select an ESP8266 or ESP32 board in Tools > Board"
#endif

// WiFi Credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// MultiSense API Server
// For local testing: e.g. "http://192.168.1.100:8000/api/arduino/stream?format=text"
// For cloud Render deployment:
const char* serverUrl = "https://multisense-ai-pslv.onrender.com/api/arduino/stream?format=text";

// Pins
const int BUZZER_PIN = 14; // D5 / GPIO14
const int LED_PIN = 2;     // Built-in LED

// Simulated/Real Sensor Variables
long irVal = 0;
float ax = 0.0, ay = 0.0, az = 0.0;
float luxVal = -2.00;

unsigned long lastTime = 0;
const unsigned long timerDelay = 2000; // Send reading every 2 seconds

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected! IP Address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  if (millis() - lastTime > timerDelay) {
    if (WiFi.status() == WL_CONNECTED) {
      WiFiClient client;
      HTTPClient http;

      // Construct sensor payload string:
      // Format:
      //   IR: 0
      //   ACC: 0, 0, 0
      //   Light: -2.00 lux
      String payload = "IR: " + String(irVal) + "\n" +
                       "ACC: " + String(ax, 2) + ", " + String(ay, 2) + ", " + String(az, 2) + "\n" +
                       "Light: " + String(luxVal, 2) + " lux";

      Serial.println("\n[HTTP] Sending sensor payload to MultiSense API...");
      Serial.println(payload);

      http.begin(client, serverUrl);
      http.addHeader("Content-Type", "text/plain");

      int httpResponseCode = http.POST(payload);

      if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.print("[HTTP] Server Response (Code ");
        Serial.print(httpResponseCode);
        Serial.println("):");
        Serial.println("  " + response);

        // Response format: PRED:ArmFlapping|CONF:0.45|RISK:MEDIUM|BUZZER:0|DB:SAVED
        if (response.indexOf("BUZZER:1") >= 0 || response.indexOf("RISK:HIGH") >= 0) {
          Serial.println("  --> [ALERT] High Stimming Risk Detected! Activating Buzzer!");
          digitalWrite(LED_PIN, HIGH);
          digitalWrite(BUZZER_PIN, HIGH);
          delay(400);
          digitalWrite(BUZZER_PIN, LOW);
        } else {
          digitalWrite(LED_PIN, LOW);
          digitalWrite(BUZZER_PIN, LOW);
        }
      } else {
        Serial.print("[HTTP] POST Error: ");
        Serial.println(httpResponseCode);
      }
      http.end();
    }
    lastTime = millis();
  }
}
