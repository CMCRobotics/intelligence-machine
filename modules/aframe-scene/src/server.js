const express = require('express');
const multer = require('multer');
const unzipper = require('unzipper');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const Paho = require('paho-mqtt'); // Added Paho MQTT client

const app = express();
const port = 3000;

app.use(cors());
app.use(express.static('dist'));

const upload = multer({ storage: multer.memoryStorage() });

// Paho MQTT client setup
const clientId = 'mqtt-server-' + Math.random().toString(16).substr(2, 8);
const client = new Paho.Client('localhost', 9001, '/mqtt', clientId); // Connect to ws://localhost:9001/mqtt

client.onConnectionLost = (responseObject) => {
  if (responseObject.errorCode !== 0) {
    console.log('Connection lost:', responseObject.errorMessage);
  }
};

client.onMessageArrived = (message) => {
  console.log(`Message arrived [${message.destinationName}]: ${message.payloadString}`);
};

client.connect({
  onSuccess: () => {
    console.log('Connected to MQTT broker (Paho)');
    // Subscribe to any topics if needed
  },
  onFailure: (responseObject) => {
    console.error('Failed to connect to MQTT broker (Paho):', responseObject.errorMessage);
  },
  useSSL: false, // Assuming ws://, not wss://
  // For Paho, the WebSocket path is typically /mqtt, ensure your broker supports it
});

app.post('/upload', upload.single('model'), (req, res) => {
  const { name, 'team-id': teamId } = req.body;
  const file = req.file;

  if (!name || !teamId || !file) {
    return res.status(400).send('Missing name, team-id, or file.');
  }

  const modelName = `${teamId}-${name}`;
  const modelPath = path.join(__dirname, '..', 'public', 'models', modelName);

  fs.mkdirSync(modelPath, { recursive: true });

  const stream = unzipper.Extract({ path: modelPath });
  stream.on('finish', () => {
    const metadataFilePath = path.join(modelPath, 'metadata.json');
    let classCount = 0;
    try {
      const metadataContent = fs.readFileSync(metadataFilePath, 'utf-8');
      const metadata = JSON.parse(metadataContent);
      if (metadata.labels && Array.isArray(metadata.labels)) {
        classCount = metadata.labels.length;
      }
    } catch (error) {
      console.error('Could not read or parse metadata.json:', error);
    }

    const modelUrl = `/models/${modelName}/model.json`;
    const metadataUrl = `/models/${modelName}/metadata.json`;
    const modelType = 'image'; // Assuming image models for now

    const homieBaseTopic = `homie/${teamId}/model-${name}`;
    const properties = {
      'modelUrl': modelUrl,
      'metadataUrl': metadataUrl,
      'type': modelType,
      'timestamp': new Date().toISOString(),
      'modelName': name,
      'terminalId': req.body.terminalId || 'unknown',
      'classCount': classCount,
      'storagePath': modelPath
    };

    for (const [key, value] of Object.entries(properties)) {
      const message = new Paho.Message(value.toString());
      message.destinationName = `${homieBaseTopic}/${key}`;
      message.retained = true;
      client.send(message);
    }

    res.status(200).send({ message: 'Model uploaded and advertised successfully.' });
  });

  stream.write(file.buffer);
  stream.end();
});

app.listen(port, '0.0.0.0', () => {
  console.log(`Server listening at http://0.0.0.0:${port}`);
});
