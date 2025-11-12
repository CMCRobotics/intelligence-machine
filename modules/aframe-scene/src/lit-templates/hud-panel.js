import { LitElement, html, css } from 'lit';

class HudPanel extends LitElement {
  static properties = {
    visible: { type: Boolean },
  };

  constructor() {
    super();
    this.visible = true; // Initially hidden
  }

  static styles = css`
    :host {
      display: block; /* Hidden by default */
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%); /* Center the element */
      width: 60%; /* Adjust width as needed */
      height: 40%; /* Adjust height as needed */
      background-color: rgba(245, 122, 122, 0.8); /* Semi-transparent black background */
      pointer-events: auto; /* Allow pointer events to interact with the HUD */
      z-index: 1000; /* Ensure it's on top of other content */
      border-radius: 15px; /* Add rounded corners */
    }

    :host([visible]) {
      display: block; /* Show when visible attribute is present */
    }

    .hud-content {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      color: white;
      font-size: 24px;
      text-align: center;
    }
  `;

  render() {
    return html`
      <div class="hud-content">
        <h1>HUD Panel Active</h1>
      </div>
    `;
  }

  show() {
    this.visible = true;
    // Suppress interactions with the A-Frame scene
    const aframeCanvas = document.querySelector('a-scene')?.canvas;
    if (aframeCanvas) {
      aframeCanvas.style.pointerEvents = 'none';
    }
  }

  hide() {
    this.visible = false;
    // Restore interactions with the A-Frame scene
    const aframeCanvas = document.querySelector('a-scene')?.canvas;
    if (aframeCanvas) {
      aframeCanvas.style.pointerEvents = 'auto';
    }
  }
}

customElements.define('hud-panel', HudPanel);

// Export functions to control the HUD
export { HudPanel };
