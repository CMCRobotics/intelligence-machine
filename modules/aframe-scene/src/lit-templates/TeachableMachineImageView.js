import { LitElement, html, css } from 'lit';
import * as tmImage from '@teachablemachine/image';
import { createMqttHomieObserver } from '@cmcrobotics/homie-lit';
import { merge } from 'rxjs';

class TeachableMachineImageView extends LitElement {
  static get properties() {
    return {
      name: { type: String },
      predictions: { type: Array },
      deviceId: { type: String },
      testCountdown: { type: Number },
      overallCountdown: { type: Number },
    };
  }

  constructor() {
    super();
    this.predictions = [];
    this.modelURL = null;
    this.metadataURL = null;
    this.modelType = null;
    this.isTesting = false;
    this.testParameters = null;
    this.testStartTime = null;
    this.flashingInterval = null;
    this.isLabelFlashing = false;
    this.countdownInterval = null;
    this.overallCountdownInterval = null;
  }

  connectedCallback() {
    super.connectedCallback();
    this.deviceId = localStorage.getItem('deviceId');
    if (this.deviceId) {
      this.connect();
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this.flashingInterval) clearInterval(this.flashingInterval);
    if (this.countdownInterval) clearInterval(this.countdownInterval);
    if (this.overallCountdownInterval) clearInterval(this.overallCountdownInterval);
  }

  connect() {
    if (!this.homieObserver) {
      const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const mqttUrl = (window.APP_CONFIG && window.APP_CONFIG.MQTT_BROKER_URL) || `${scheme}://${window.location.hostname}:9001`;
      this.homieObserver = createMqttHomieObserver(mqttUrl);
      const topic = `homie/terminal-${this.deviceId}/activeModel/#`;

      merge(
        this.homieObserver.created$,
        this.homieObserver.updated$
      ).subscribe(event => {
        if (event.type === 'property') {
          const propertyId = event.property.id;
          const value = event.property.value;

          switch (propertyId) {
            case 'set':
              try {
                const modelData = JSON.parse(value);
                this.modelName = modelData.name;
                this.uploaderTeamId = modelData.uploaderTeamId;
                this.modelType = modelData.type;

                if (!this.isTesting) {
                  const modelBaseName = `${this.uploaderTeamId}-${this.modelName}`;
                  this.modelURL = `/models/${modelBaseName}/model.json`;
                  this.metadataURL = `/models/${modelBaseName}/metadata.json`;
                  this.init();
                }
              } catch (e) {
                console.error('Invalid model data payload:', e);
              }
              break;
            case 'test':
              this.handleTestRequest(value);
              break;
          }
        }
      });

      this.homieObserver.subscribe(topic);
    }
  }

  handleTestRequest(payload) {
    try {
      const params = JSON.parse(payload);
      if (params.duration !== undefined && params.confidence !== undefined && params.class !== undefined && params.overallTimeout !== undefined) {
        
        // Defensive Check: Model Readiness
        if (!this.model) {
          console.warn('Test requested but model is not yet loaded.');
          return;
        }

        // Handle Random Class Selection
        if (params.class === -1) {
          params.class = Math.floor(Math.random() * this.maxPredictions);
          console.log(`Randomly selected class index: ${params.class}`);
        }

        // Defensive Check: Class Index Validity
        if (params.class < 0 || params.class >= this.maxPredictions) {
          console.error(`Test requested for invalid class index: ${params.class}. Max index is ${this.maxPredictions - 1}`);
          return;
        }

        if (this.flashingInterval) clearInterval(this.flashingInterval);
        if (this.countdownInterval) clearInterval(this.countdownInterval);
        if (this.overallCountdownInterval) clearInterval(this.overallCountdownInterval);

        this.testParameters = params;
        this.isTesting = true;
        this.testStartTime = null;
        this.testCountdown = null;
        this.isLabelFlashing = false;
        
        console.log('Starting confidence test:', this.testParameters);

        this.flashingInterval = setInterval(() => {
          this.isLabelFlashing = !this.isLabelFlashing;
          this.requestUpdate();
        }, 500);

        const overallEndTime = Date.now() + this.testParameters.overallTimeout;
        this.overallCountdownInterval = setInterval(() => {
          const remaining = overallEndTime - Date.now();
          this.overallCountdown = Math.ceil(remaining / 1000);
          if (remaining <= 0) {
            this.overallCountdown = 0;
            clearInterval(this.overallCountdownInterval);
            this.publishTestResult(false);
            this.isTesting = false;
          }
        }, 100);
      }
    } catch (e) {
      console.error('Invalid test payload:', e);
    }
  }

  async init() {
    if (this.model) return; // Already initialized

    this.model = await tmImage.load(this.modelURL, this.metadataURL);
    this.maxPredictions = this.model.getTotalClasses();

    this.webcam = new tmImage.Webcam(200, 200, true);
    await this.webcam.setup();
    await this.webcam.play();
    window.requestAnimationFrame(this.loop.bind(this));

    const container = this.shadowRoot.getElementById('webcam-container');
    if (container) {
        container.innerHTML = '';
        container.appendChild(this.webcam.canvas);
    }
  }

  async loop() {
    if (this.webcam) {
        this.webcam.update();
        await this.predict();
    }
    window.requestAnimationFrame(this.loop.bind(this));
  }

  async predict() {
    if (this.model) {
        const prediction = await this.model.predict(this.webcam.canvas);
        this.predictions = prediction;
        this.checkConfidenceTest();
    }
  }

  checkConfidenceTest() {
    if (!this.isTesting || !this.testParameters) return;

    const { duration, confidence, class: classIndex } = this.testParameters;

    const prediction = this.predictions[classIndex];

    if (prediction && (prediction.probability * 100) >= confidence) {
      if (this.testStartTime === null) {
        this.testStartTime = Date.now();
        this.testCountdown = Math.ceil(duration / 1000);
        if(this.countdownInterval) clearInterval(this.countdownInterval);
        this.countdownInterval = setInterval(() => {
            const elapsed = Date.now() - this.testStartTime;
            const remaining = duration - elapsed;
            this.testCountdown = Math.ceil(remaining / 1000);
            if (remaining <= 0) {
                this.testCountdown = 0;
                clearInterval(this.countdownInterval);
            }
        }, 100);
      }

      if (Date.now() - this.testStartTime >= duration) {
        this.publishTestResult(true);
        this.isTesting = false;
        this.testParameters = null;
        if (this.flashingInterval) clearInterval(this.flashingInterval);
        if (this.countdownInterval) clearInterval(this.countdownInterval);
        if (this.overallCountdownInterval) clearInterval(this.overallCountdownInterval);
        this.isLabelFlashing = false;
        this.testCountdown = null;
      }
    } else {
      this.testStartTime = null; // Reset timer if confidence drops
      this.testCountdown = null;
      if (this.countdownInterval) {
        clearInterval(this.countdownInterval);
        this.countdownInterval = null;
      }
    }
  }

  publishTestResult(success) {
    if (success) {
        const resultTopic = `terminal-${this.deviceId}/activeModel/testSuccess`;
        const payload = JSON.stringify({ ...this.testParameters, timestamp: Date.now(), team: localStorage.getItem('teamId') });
        this.homieObserver.publish(resultTopic, payload);
        console.log(`Test succeeded. Publishing result to ${resultTopic}.`);
    } else {
        const resultTopic = `terminal-${this.deviceId}/activeModel/test-result`;
        const payload = JSON.stringify({ success });
        this.homieObserver.publish(resultTopic, payload);
        console.log(`Test failed. Publishing result.`);
    }
  }

  render() {
    return html`
      <div class="view">
        <h2>${this.name}</h2>
        ${this.isTesting ? html`<h3>Overall Test Ends In: ${this.overallCountdown}s</h3>` : ''}
        <div id="webcam-container"></div>
        <div id="label-container">
          ${this.predictions.map(
            (prediction, index) => {
                const isTestingClass = this.isTesting && this.testParameters && this.testParameters.class === index;
                const flashingClass = isTestingClass && this.isLabelFlashing ? 'flashing' : '';
                return html`
              <div class="prediction ${flashingClass}">
                ${prediction.className}: ${prediction.probability.toFixed(2)}
                ${isTestingClass && this.testCountdown !== null ? ` - Countdown: ${this.testCountdown}` : ''}
              </div>
            `}
          )}
        </div>
      </div>
    `;
  }

  static styles = css`
    .view {
      padding: 20px;
      border: 1px dashed green;
      background-color: #e8f5e9;
      text-align: center;
      width: 80%;
      height: 80%;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
    }
    .prediction.flashing {
      background-color: yellow;
    }
  `;
}

customElements.define('teachable-machine-image-view', TeachableMachineImageView);

export { TeachableMachineImageView };
