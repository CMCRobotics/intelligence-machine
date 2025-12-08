const Paho = require("paho-mqtt"); // Added Paho MQTT client

const deviceId = process.argv[2];

if (!deviceId) {
  console.error("Usage: bun set-model.js <deviceId>"); // Updated usage
  process.exit(1);
}

const baseUrl = "http://localhost:9000/models/sign-language";
const modelUrl = `${baseUrl}/model.json`;
const metadataUrl = `${baseUrl}/metadata.json`;

// Paho MQTT client setup for WebSocket
const clientId = "mqtt-set-model-" + Math.random().toString(16).substr(2, 8);
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

    const topicPrefix = `homie/terminal-${deviceId}/activeModel`;

    const publishMessage = (topic, payload) => {
        const message = new Paho.Message(payload);
        message.destinationName = topic;
        message.retained = true;
        client.send(message);
    };

    publishMessage(`${topicPrefix}/model-url`, modelUrl);
    publishMessage(`${topicPrefix}/metadata-url`, metadataUrl);
    publishMessage(`${topicPrefix}/type`, "image");

    console.log("Model sign-language information published");
    client.disconnect();
  },
  onFailure: (responseObject) => {
    console.error("Failed to connect to MQTT broker (Paho):");
  },
  useSSL: false, // Assuming ws://, not wss://
});
