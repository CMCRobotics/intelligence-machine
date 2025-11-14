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
        <h2>Team: ${this.teamName}</h2>
        <p>Please wait for the next instruction.</p>
      </div>
    `;
  }

  static styles = css`
    .view {
      padding: 20px;
      border: 1px dashed #ccc;
      background-color: #f5f5f5;
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

customElements.define('waiting-view', WaitingView);

export { WaitingView };
