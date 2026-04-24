import { LitElement, html, css } from 'lit';

class WaitingView extends LitElement {
  static properties = {
    teamName: { type: String },
  };

  constructor() {
    super();
    this.teamName = '';
  }

  connectedCallback() {
    super.connectedCallback();
    this.teamName = localStorage.getItem('teamName');
  }

  render() {
    return html`
      <div class="view">
        <div class="team-badge" style="background-color: ${this.teamName.toLowerCase()}; color: ${this._getTextColor(this.teamName)}">
          Team ${this.teamName}
        </div>
        <h2>Waiting...</h2>
        <div class="spinner"></div>
        <p>Please wait for the next instructions.</p>
      </div>
    `;
  }

  _getTextColor(teamName) {
    if (!teamName) return 'white';
    const darkColors = ['blue', 'red', 'purple', 'green'];
    return darkColors.includes(teamName.toLowerCase()) ? 'white' : 'black';
  }

  static styles = css`
    :host {
      display: block;
      width: 100%;
      height: 100%;
    }

    .view {
      padding: 40px;
      text-align: center;
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      box-sizing: border-box;
      color: white;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .team-badge {
      padding: 8px 30px;
      border-radius: 20px;
      font-weight: bold;
      font-size: 2rem;
      margin-bottom: 30px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.3);
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    h2 {
      margin: 0;
      font-size: 2.5rem;
      text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
      letter-spacing: 2px;
    }

    p {
      font-size: 1.2rem;
      color: #ccc;
      margin-top: 20px;
    }

    .spinner {
      width: 50px;
      height: 50px;
      border: 5px solid rgba(255, 255, 255, 0.1);
      border-top: 5px solid white;
      border-radius: 50%;
      margin: 20px 0;
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  `;
}

customElements.define('waiting-view', WaitingView);

export { WaitingView };
