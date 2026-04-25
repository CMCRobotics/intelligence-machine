const mqtt = require('mqtt');
const mqttBrokerUrl = process.env.MQTT_BROKER_URL || 'mqtt://localhost:1883';
const client = mqtt.connect(mqttBrokerUrl);

const topicPattern = process.argv[2] || '#';

console.log(`Connecting to MQTT broker at ${mqttBrokerUrl}...`);
console.log(`Watching topic pattern: ${topicPattern}`);

client.on('connect', () => {
    console.log('Connected to MQTT broker');
    client.subscribe(topicPattern, (err) => {
        if (err) {
            console.error(`Failed to subscribe to ${topicPattern}:`, err);
            process.exit(1);
        }
        console.log(`Subscribed to ${topicPattern}`);
    });
});

client.on('message', (topic, message, packet) => {
    const payload = message.toString();
    const retained = packet.retain ? '[RETAINED]' : '';
    console.log(`${retained} ${topic}: ${payload}`);
});

client.on('error', (err) => {
    console.error('MQTT Error:', err);
});

process.on('SIGINT', () => {
    console.log('\nClosing connection...');
    client.end();
    process.exit();
});
