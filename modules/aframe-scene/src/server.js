const express = require('express');
const multer = require('multer');
const unzipper = require('unzipper');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const mqtt = require('mqtt');

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());

const upload = multer({ storage: multer.memoryStorage() });
const mqttBrokerUrl = process.env.MQTT_BROKER_URL || 'ws://localhost:9001';
const client = mqtt.connect(mqttBrokerUrl);

app.get('/config.js', (req, res) => {
  const config = {
    MQTT_BROKER_URL: process.env.MQTT_BROKER_URL_CLIENT || `ws://${req.hostname}:9001`,
    API_URL: process.env.API_URL || `http://${req.hostname}:${port}`
  };
  res.type('application/javascript');
  res.send(`window.APP_CONFIG = ${JSON.stringify(config)};`);
});

client.on('connect', () => {
  console.log('Connected to MQTT broker');
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
      client.publish(`${homieBaseTopic}/${key}`, value.toString(), { retain: true });
    }

    res.status(200).send({ message: 'Model uploaded and advertised successfully.' });
  });

  stream.write(file.buffer);
  stream.end();
});

app.listen(port, '0.0.0.0', () => {
  console.log(`Server listening at http://0.0.0.0:${port}`);
});
