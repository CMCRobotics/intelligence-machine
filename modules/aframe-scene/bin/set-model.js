const mqtt = require('mqtt');
const mqttBrokerUrl = process.env.MQTT_BROKER_URL || 'mqtt://localhost:1883';
const client = mqtt.connect(mqttBrokerUrl);

const deviceId = process.argv[2];

if (!deviceId) {
  console.error('Usage: node set-model.js <deviceId>');
  process.exit(1);
}

const baseUrl = 'http://localhost:9000/models/sign-language';
const modelUrl = `${baseUrl}/model.json`;
const metadataUrl = `${baseUrl}/metadata.json`;

client.on('connect', () => {
  console.log('Connected to MQTT broker');

  const topicPrefix = `homie/terminal-${deviceId}/activeModel`;

  client.publish(`${topicPrefix}/model-url`, modelUrl, { retain: true });
  client.publish(`${topicPrefix}/metadata-url`, metadataUrl, { retain: true });
  client.publish(`${topicPrefix}/type`, 'image', { retain: true });

  console.log('Model sign-language information published');
  client.end();
});
