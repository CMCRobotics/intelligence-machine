const Paho = require("paho-mqtt"); // Added Paho MQTT client

const deviceId = process.argv[2];
const classIndex = parseInt(process.argv[3], 10);
const duration = parseInt(process.argv[4], 10);
const confidence = parseInt(process.argv[5], 10);

if (!deviceId || isNaN(classIndex) || isNaN(duration) || isNaN(confidence)) {
  console.error("Usage: bun trigger-test.js <deviceId> <classIndex> <durationMs> <confidencePercent>"); // Updated usage
  process.exit(1);
}

const payload = JSON.stringify({
  class: classIndex,
  duration: duration,
  confidence: confidence,
});

// Paho MQTT client setup for WebSocket
const clientId = "mqtt-trigger-test-" + Math.random().toString(16).substr(2, 8);
const client = new Paho.Client("localhost", 9001, "/mqtt", clientId); // Connect to ws://localhost:9001/mqtt

client.onConnectionLost = (responseObject) => {
  if (responseObject.errorCode !== 0) {
    console.log("Connection lost:", responseObject.errorMessage);
  }
};

client.onMessageArrived = (message) => {
  console.log(`Message arrived [${message.destinationName}]: ${message.payloadString}`);
};

client.connect({
  onSuccess: () => {
    console.log("Connected to MQTT broker (Paho)");

    const topic = `homie/terminal-${deviceId}/activeModel/test`;
    const message = new Paho.Message(payload);
    message.destinationName = topic;
    client.send(message);

    console.log(`Test triggered with payload: ${payload}`);
    client.disconnect();
  },
  onFailure: (responseObject) => {
    console.error("Failed to connect to MQTT broker (Paho):");
  },
  useSSL: false, // Assuming ws://, not wss://
});
