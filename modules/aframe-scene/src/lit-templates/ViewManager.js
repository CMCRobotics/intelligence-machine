import { LitElement, html, css } from 'lit';
import { ExampleView1 } from './ExampleView1.js';
import { ExampleView2 } from './ExampleView2.js'; 
import { createMqttHomieObserver, setLogLevel } from '@cmcrobotics/homie-lit';

let VIEWS = [
       { name: 'example1', tagName: 'example-view-1' }
      ,{ name: 'example2', tagName: 'example-view-2' }];



class ViewManager extends LitElement {
  static properties = {
    views: { type: Array }, // Array of { name: string, tagName: string }
    activeViewName: { type: String },
    modelURL: {type: String},
    metadataURL: {type: String},
    deviceId: {type: String}
  };

  
  constructor() {
    super();
    this.views = VIEWS;
    this.activeViewName = VIEWS[0].name;
    this._currentViewElement = null; // To keep track of the currently rendered element

    // setLogLevel('debug'); // Set Homie-lit log level
    
  }

  connect() {
    console.log('connect() called'); // Added log to track calls
    if (!this.homieObserver) {
      console.log('this.homieObserver is undefined, attempting to create...'); // Added log
      try {
        this.homieObserver = createMqttHomieObserver("ws://localhost:9001");
        console.log('Value of this.homieObserver after assignment:', this.homieObserver); // Added log

        if (this.homieObserver) {
          this.homieObserver.updated$.subscribe(
            (event) => {
              if (event.type == 'property') {
                if (event.device.id === 'terminal-' + this.deviceId && event.node.id === 'ui-control' && event.property.id === 'switch') {
                  // TODO : Check if view is already known
                  this.activeViewName = event.property.value;
                }
              }
            },
            (error) => {
              console.error('Error in subscription:', error);
            }
          );

          this.homieObserver.subscribe("terminal-" + this.deviceId + "/#"); 
        } else {
          console.error('createMqttHomieObserver returned undefined or null.'); // Added log for failure
        }
      } catch (error) {
        console.error('Error during createMqttHomieObserver or subscription:', error); // Added error handling
      }
    }
  }

  /**
   * Switches the active view to the one specified by viewName.
   * @param {string} viewName - The name of the view to switch to.
   */
  switchView(viewName) {
    this.activeViewName = viewName;
  }

  // Use updated lifecycle callback to manage DOM updates
  updated(changedProperties) {
    if (changedProperties.has('activeViewName')) {
      this._updateView();
    }
  }

  _updateView() {
    // Remove the previous view element if it exists
    if (this._currentViewElement) {
      this._currentViewElement.remove();
      this._currentViewElement = null;
    }

    const activeViewConfig = this.views.find(view => view.name === this.activeViewName);
    const container = this.shadowRoot.querySelector('.view-manager');

    if (!container) {
      // If the container is not yet available, schedule the update for later.
      // This might happen if _updateView is called before the render method has fully attached the DOM.
      requestAnimationFrame(() => this._updateView());
      return;
    }

    if (activeViewConfig) {
      // Create the new view element using plain JavaScript DOM API
      const newViewElement = document.createElement(activeViewConfig.tagName);
      container.appendChild(newViewElement);
      this._currentViewElement = newViewElement;
    } else {
      // If no active view is found, ensure the container is empty
      container.innerHTML = ''; // Clear any existing content
    }
  }

  render() {
    // The render method now only sets up the container.
    // The actual view element will be managed by _updateView.
    return html`
      <div class="view-manager">
        <!-- The view element will be appended here by _updateView -->
      </div>
    `;
  }

  connectedCallback() {
    super.connectedCallback(); // Ensure LitElement's connectedCallback is called
    this.connect();
  }

  static styles = css`
    .view-manager {
      width: 100%;
      height: 100%;
      display: flex;
      justify-content: center;
      align-items: center;
      border: 1px solid #ccc; /* For visualization */
      box-sizing: border-box;
    }
  `;
}

customElements.define('view-manager', ViewManager);

export { ViewManager };
