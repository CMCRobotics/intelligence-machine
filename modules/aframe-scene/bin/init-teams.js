const Paho = require("paho-mqtt"); // Added Paho MQTT client

const TEAMS = [
    {
        id: "team-blue",
        name: "Blue"
    },
    {
        id: "team-red",
        name: "Red"
    },
    {
        id: "team-white",
        name: "White"
    }
];

// Paho MQTT client setup
const clientId = "mqtt-init-teams-" + Math.random().toString(16).substr(2, 8);
const client = new Paho.Client("localhost", 9001, "/mqtt", clientId);

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
    console.log("Publishing teams...");
    for (const team of TEAMS) {
        const publishMessage = (topic, payload) => {
            const message = new Paho.Message(payload);
            message.destinationName = topic;
            message.retained = true;
            client.send(message);
        };

        publishMessage(`homie/${team.id}/$homie`, "4.0.0");
        publishMessage(`homie/${team.id}/$name`, team.name);
        publishMessage(`homie/${team.id}/$state`, "ready");
        publishMessage(`homie/${team.id}/$nodes`, "info");
        publishMessage(`homie/${team.id}/info/$name`, "Info");
        publishMessage(`homie/${team.id}/info/$properties`, "id,name");
        publishMessage(`homie/${team.id}/info/id`, team.id);
        publishMessage(`homie/${team.id}/info/name`, team.name);
    }
    console.log("Done.");
    client.disconnect(); // Use disconnect instead of client.end() for Paho
  },
  onFailure: (responseObject) => {
    console.error("Failed to connect to MQTT broker (Paho):");
  },
  useSSL: false,
});
