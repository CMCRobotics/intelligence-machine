const mqtt = require('mqtt');

const client = mqtt.connect('ws://localhost:9001');

const TEAMS = [
    {
        id: 'team-blue',
        name: 'Blue'
    },
    {
        id: 'team-red',
        name: 'Red'
    },
    {
        id: 'team-white',
        name: 'White'
    }
]

client.on('connect', () => {
    console.log('Connected to MQTT broker');
    console.log('Publishing teams...');
    for (const team of TEAMS) {
        client.publish(`homie/${team.id}/$homie`, '4.0.0', { retain: true });
        client.publish(`homie/${team.id}/$name`, team.name, { retain: true });
        client.publish(`homie/${team.id}/$state`, 'ready', { retain: true });
        client.publish(`homie/${team.id}/$nodes`, 'info', { retain: true });
        client.publish(`homie/${team.id}/info/$name`, 'Info', { retain: true });
        client.publish(`homie/${team.id}/info/$properties`, 'id,name', { retain: true });
        client.publish(`homie/${team.id}/info/id`, team.id, { retain: true });
        client.publish(`homie/${team.id}/info/name`, team.name, { retain: true });
    }
    console.log('Done.');
    client.end();
});
