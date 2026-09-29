const mqtt = require('mqtt');
const mqttBrokerUrl = process.env.MQTT_BROKER_URL || 'mqtt://localhost:1883';
const client = mqtt.connect(mqttBrokerUrl);

const TEAMS = [
    {
        id: 'team-blue',
        name: 'Blue',
        score: '0'
    },
    {
        id: 'team-red',
        name: 'Red',
        score: '0'
    },
    {
        id: 'team-white',
        name: 'White',
        score: '0'
    },
    {
        id: 'team-yellow',
        name: 'Yellow',
        score: '0'
    },
    {
        id: 'team-purple',
        name: 'Purple',
        score: '0'
    },
    {
        id: 'team-green',
        name: 'Green',
        score: '0'
    }
]

client.on('connect', () => {
    console.log('Connected to MQTT broker');
    console.log('Scanning for existing model properties to clear...');

    // Subscribe to all topics to find and clear existing models
    client.subscribe('homie/#');

    const clearedTopics = new Set();
    
    const onMessage = (topic, message, packet) => {
        // We look for model nodes under teams: homie/team-xxx/model-yyy/...
        if (packet.retain && topic.includes('/model-')) {
            console.log(`Clearing existing model property: ${topic}`);
            client.publish(topic, '', { retain: true });
            clearedTopics.add(topic);
        }
    };

    client.on('message', onMessage);

    // Give it a short time to clear models before publishing teams
    setTimeout(() => {
        client.removeListener('message', onMessage);
        client.unsubscribe('homie/team-#');

        console.log('Publishing teams...');
        for (const team of TEAMS) {
            client.publish(`homie/${team.id}/$homie`, '4.0.0', { retain: true });
            client.publish(`homie/${team.id}/$name`, team.name, { retain: true });
            client.publish(`homie/${team.id}/$state`, 'ready', { retain: true });
            client.publish(`homie/${team.id}/$nodes`, 'info', { retain: true });
            client.publish(`homie/${team.id}/info/$name`, 'Info', { retain: true });
            client.publish(`homie/${team.id}/info/$properties`, 'id,name,score', { retain: true });
            client.publish(`homie/${team.id}/info/id`, team.id, { retain: true });
            client.publish(`homie/${team.id}/info/name`, team.name, { retain: true });
            client.publish(`homie/${team.id}/info/score`, team.score, { retain: true });
        }
        console.log('Done.');
        client.end();
    }, 2000);
});
