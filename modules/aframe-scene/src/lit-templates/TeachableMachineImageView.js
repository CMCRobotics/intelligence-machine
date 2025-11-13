import { LitElement, html, css } from 'lit';
import * as tmImage from '@teachablemachine/image';

class TeachableMachineImageView extends LitElement {
  static get properties() {
    return {
      metadataURL: { type: String },
      modelURL: { type: String },
      name: { type: String },
      predictions: { type: Array },
    };
  }

  constructor() {
    super();
    this.predictions = [];
  }

  async firstUpdated() {
    this.init();
  }

  async init() {
    const modelURL = this.modelURL;
    const metadataURL = this.metadataURL;

    this.model = await tmImage.load(modelURL, metadataURL);
    this.maxPredictions = this.model.getTotalClasses();

    this.webcam = new tmImage.Webcam(200, 200, true); // width, height, flip
    await this.webcam.setup();
    await this.webcam.play();
    window.requestAnimationFrame(this.loop.bind(this));

    this.shadowRoot.getElementById('webcam-container').appendChild(this.webcam.canvas);
  }

  async loop() {
    this.webcam.update();
    await this.predict();
    window.requestAnimationFrame(this.loop.bind(this));
  }

  async predict() {
    const prediction = await this.model.predict(this.webcam.canvas);
    this.predictions = prediction;
  }

  render() {
    return html`
      <div class="view">
        <h2>${this.name}</h2>
        <div id="webcam-container"></div>
        <div id="label-container">
          ${this.predictions.map(
            (prediction) => html`
              <div class="prediction">
                ${prediction.className}: ${prediction.probability.toFixed(2)}
              </div>
            `
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
  `;
}

customElements.define('teachable-machine-image-view', TeachableMachineImageView);

export { TeachableMachineImageView };
