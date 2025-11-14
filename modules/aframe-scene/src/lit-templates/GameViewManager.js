import { LitElement, html, css } from 'lit';
import './TerminalView.js';
import './TeamView.js';
import './TerminalViewSwitcher.js';

const VIEWS = [
  { name: 'terminal-view', tagName: 'terminal-view' },
  { name: 'team-view', tagName: 'team-view' },
  { name: 'terminal-view-switcher', tagName: 'terminal-view-switcher' },
];

class GameViewManager extends LitElement {
  static properties = {
    views: { type: Array },
    activeViewName: { type: String },
  };

  constructor() {
    super();
    this.views = VIEWS;
    this.activeViewName = '';
    this._currentViewElement = null;
  }

  switchView(viewName) {
    this.activeViewName = viewName;
  }

  updated(changedProperties) {
    if (changedProperties.has('activeViewName')) {
      this._updateView();
    }
  }

  _updateView() {
    if (this._currentViewElement) {
      this._currentViewElement.remove();
      this._currentViewElement = null;
    }

    const activeViewConfig = this.views.find(view => view.name === this.activeViewName);
    const container = this.shadowRoot.querySelector('.view-manager');

    if (!container) {
      requestAnimationFrame(() => this._updateView());
      return;
    }

    if (activeViewConfig) {
      const newViewElement = document.createElement(activeViewConfig.tagName);
      container.appendChild(newViewElement);
      this._currentViewElement = newViewElement;
    } else {
      container.innerHTML = '';
    }
  }

  render() {
    return html`
      <div class="game-view-container">
        <div class="navigation tabs">
          <button class="${this.activeViewName === 'terminal-view' ? 'active' : ''}" @click=${() => this.switchView('terminal-view')}>Terminal View</button>
          <button class="${this.activeViewName === 'team-view' ? 'active' : ''}" @click=${() => this.switchView('team-view')}>Team View</button>
          <button class="${this.activeViewName === 'terminal-view-switcher' ? 'active' : ''}" @click=${() => this.switchView('terminal-view-switcher')}>Terminal Switcher</button>
        </div>
        <div class="view-manager">
        </div>
      </div>
    `;
  }

  connectedCallback() {
    super.connectedCallback();
    if (this.views.length > 0) {
        this.activeViewName = this.views[0].name;
    }
  }

  static styles = css`
    .game-view-container {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding-top: 20px;
    }

    .navigation {
      margin-bottom: 20px;
    }

    .tabs {
      display: flex;
      background-color: rgba(0, 0, 0, 0.5);
      border-radius: 10px;
      padding: 5px;
    }

    .tabs button {
      padding: 10px 20px;
      cursor: pointer;
      border: none;
      background-color: transparent;
      color: white;
      border-radius: 8px;
    }

    .tabs button.active {
      background-color: #4CAF50;
    }

    .view-manager {
      width: 100%;
      display: flex;
      justify-content: center;
      align-items: center;
      box-sizing: border-box;
    }
  `;
}

customElements.define('game-view-manager', GameViewManager);

export { GameViewManager };
