import { LitElement, html, css } from 'lit';
import { ViewManager } from './ViewManager.js';
import { SessionManager } from './SessionManager.js';
// Example views are now managed by ViewManager using tag names, so direct imports are not needed here.

class HudPanel extends LitElement {
  static properties = {
    visible: { type: Boolean },
    views: { type: Array }, // Array of { name: string, tagName: string }
    activeViewName: { type: String },
    deviceId: { type: String },
    teamColor: { type: String },
  };

  constructor() {
    super();
    this.visible = true; // Initially hidden
    this.deviceId = localStorage.getItem('deviceId');
    this.teamColor = 'gray'; // Default color
    this._handleSessionUpdate = this._handleSessionUpdate.bind(this);
  }

  connectedCallback() {
    super.connectedCallback();
    this.addEventListener('session-updated', this._handleSessionUpdate);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.removeEventListener('session-updated', this._handleSessionUpdate);
  }

  _handleSessionUpdate(event) {
    const { selectedTeam } = event.detail;
    if (selectedTeam && selectedTeam.id) {
      const color = selectedTeam.id.split('-')[1] || 'gray';
      this.teamColor = color;
    } else {
      this.teamColor = 'gray';
    }
  }

  updated(changedProperties) {
    if (changedProperties.has('teamColor')) {
      const teamColorMapping = {
        red: '175, 0, 0',
        blue: '0, 0, 175',
        green: '0, 175, 0',
        yellow: '175, 175, 0',
        white: '255, 255, 255',
        gray: '128, 128, 128'
      };
      const rgb = teamColorMapping[this.teamColor] || teamColorMapping['gray'];
      this.style.setProperty('--team-color-rgb', rgb);
    }
  }

  firstUpdated() {
    const sessionManager = this.shadowRoot.querySelector('#session-manager');
    const viewManager = this.shadowRoot.querySelector('view-manager');
    if (viewManager) {
      viewManager.sessionManager = sessionManager;
    }
  }
  
  static styles = css`
    :host {
      display: block;
      position: fixed;
      top: 50%;
      left: 65%;
      transform: translate(-50%, -50%); /* Center the element */
      width: 65%;
      height: 90%;
      background-color: rgba(0, 0, 0, 0.7);
      pointer-events: auto; /* Allow pointer events to interact with the HUD */
      z-index: 1000; /* Ensure it's on top of other content */
      border-radius: 15px; /* Add rounded corners */
      --team-color-rgb: 128, 128, 128; /* Default to gray */
      border: 15px solid rgba(var(--team-color-rgb), 1);
      box-shadow: 0 0 15px 5px rgba(var(--team-color-rgb), 0.9);
      display: flex; /* Use flexbox for layout */
      flex-direction: column; /* Stack children vertically */
      align-items: center; /* Center horizontally */
    }

    :host([visible]) {
      display: block; /* Show when visible attribute is present */
    }

    .hud-content {
      color: #ffffffbc;
      font-size: 24px;
      text-align: center;
      height: 100%;
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start; /* Align content to the top */
      padding-top: 20px; /* Add some padding at the top */
    }

    
    .view-container {
      flex-grow: 1; /* Allow view-container to take available space */
      width: 95%; /* Make view container take up most of the width */
      height: calc(100% - 60px); /* Adjust height to account for controls and padding */
      box-sizing: border-box;
      display: flex;
      justify-content: center;
      align-items: center;
      color: black;
    }

    /* Ensure the view-manager itself takes up the space */
    view-manager {
      width: 100%;
      height: 100%;
    }

  `;

  render() {
    return html`
      <div class="hud-content">

        <session-manager id="session-manager"></session-manager>

        <div class="view-container">
          <view-manager 
            .deviceId=${this.deviceId}
            .activeViewName="teachable-machine-image"
          ></view-manager>
        </div>
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
