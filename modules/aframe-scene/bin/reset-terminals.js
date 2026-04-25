const mqtt = require('mqtt');
const mqttBrokerUrl = process.env.MQTT_BROKER_URL || 'mqtt://localhost:1883';
const client = mqtt.connect(mqttBrokerUrl);

console.log(`Connecting to MQTT broker at ${mqttBrokerUrl}...`);

client.on('connect', () => {
    console.log('Connected to MQTT broker');
    console.log('Scanning for terminal properties to reset...');

    // We use broad wildcards to catch all subtopics. 
    // We'll subscribe to the root of homie and terminal to ensure we see everything.
    client.subscribe('homie/#');
    client.subscribe('terminal/#');
});

const clearedTopics = new Set();

client.on('message', (topic, message, packet) => {
    const payload = message.toString();
    // Debug: console.log(`Received: ${topic} (retained: ${packet.retain})`);

    if (packet.retain && !clearedTopics.has(topic)) {
        const isTerminalTopic = topic.includes('terminal-');
        const isTerminalProperty = topic.endsWith('/terminalId') || payload.startsWith('terminal-');

        if (isTerminalTopic || isTerminalProperty) {
            console.log(`Clearing retained topic: ${topic} (content: ${payload})`);
            client.publish(topic, '', { retain: true });
            clearedTopics.add(topic);
        }
    }
});

// Give it some time to receive all retained messages and clear them
setTimeout(() => {
    if (clearedTopics.size === 0) {
        console.log('No retained terminal properties found.');
    } else {
        console.log(`Successfully cleared ${clearedTopics.size} topics.`);
    }
    console.log('Done.');
    client.end();
}, 2000);
