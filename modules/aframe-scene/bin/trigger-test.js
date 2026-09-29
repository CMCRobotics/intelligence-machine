const mqtt = require('mqtt');
const client = mqtt.connect('mqtt://localhost:1883');

const deviceId = process.argv[2];
const classIndex = parseInt(process.argv[3], 10);
const duration = parseInt(process.argv[4], 10);
const confidence = parseInt(process.argv[5], 10);

if (!deviceId || isNaN(classIndex) || isNaN(duration) || isNaN(confidence)) {
  console.error('Usage: node trigger-test.js <deviceId> <classIndex> <durationMs> <confidencePercent>');
  process.exit(1);
}

const payload = JSON.stringify({
  class: classIndex,
  duration: duration,
  confidence: confidence,
});

client.on('connect', () => {
  console.log('Connected to MQTT broker');

  const topic = `homie/terminal-${deviceId}/activeModel/test`;
  client.publish(topic, payload);

  console.log(`Test triggered with payload: ${payload}`);
  client.end();
});
