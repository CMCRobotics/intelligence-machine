import { LitElement, html, css } from 'lit';

import { TeamSelectorView } from './TeamSelectorView.js';
import { TeachableMachineImageView } from './TeachableMachineImageView.js';
import { TeachableMachineUploadView } from './TeachableMachineUploadView.js';
import { WaitingView } from './WaitingView.js';
import { createMqttHomieObserver, setLogLevel } from '@cmcrobotics/homie-lit';
import { merge } from 'rxjs';

// Define the list of views, placing the team selector first
let VIEWS = [
       { name: 'team-selector', tagName: 'team-selector-view' } // New team selector view
      ,{ name: 'waiting-view', tagName: 'waiting-view' }
      ,{ name: 'teachable-machine-image', tagName: 'teachable-machine-image-view' }
      ,{ name: 'teachable-machine-upload', tagName: 'teachable-machine-upload-view' }];



class ViewManager extends LitElement {
  static properties = {
    views: { type: Array }, // Array of { name: string, tagName: string }
    activeViewName: { type: String },
    deviceId: {type: String},
    sessionManager: { type: Object }
  };

  
  constructor() {
    super();
    this.views = VIEWS;
    // Initial active view will be determined in connectedCallback based on localStorage
    this.activeViewName = ''; 
    this._currentViewElement = null; // To keep track of the currently rendered element

    setLogLevel('debug'); // Set Homie-lit log level
    
  }

  connect() {
    if (!this.homieObserver) {
      try {
        const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const mqttUrl = (window.APP_CONFIG && window.APP_CONFIG.MQTT_BROKER_URL) || `${scheme}://${window.location.hostname}:9001`;
        this.homieObserver = createMqttHomieObserver(mqttUrl);
        
        if (this.homieObserver) {
          merge(
            this.homieObserver.created$,
            this.homieObserver.updated$
            ).subscribe(
            (event) => {
              if (event.type === 'property') {
                // Listen for view switch commands from UI control
                if (event.device.id === 'terminal-' + this.deviceId && event.node.id === 'ui-control' && event.property.id === 'switch') {
                  this.switchView(event.property.value);
                }
              }
            },
            (error) => {
              console.error('Error in subscription:', error);
            }
          );

          // Subscribe to general terminal events, specific topics handled by views
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
    // The _updateView method is called automatically by LitElement's updated lifecycle callback
    // when activeViewName changes.
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
      
      // Pass necessary properties to the new view element
      // For TeamSelectorView, we need to pass deviceId
      if (activeViewConfig.tagName === 'team-selector-view') {
        newViewElement.deviceId = this.deviceId;
        newViewElement.sessionManager = this.sessionManager;
        // Add event listener for team-selected event
        newViewElement.addEventListener('team-selected', this.handleTeamSelected.bind(this));
        newViewElement.addEventListener('team-cleared', this.handleTeamCleared.bind(this));
      }
      
      if (activeViewConfig.tagName === 'teachable-machine-image-view') {
        newViewElement.deviceId = this.deviceId;
        newViewElement.name = "Teachable Machine Image Model";
      }

      if (activeViewConfig.tagName === 'teachable-machine-upload-view') {
        newViewElement.deviceId = this.deviceId;
      }

      container.appendChild(newViewElement);
      this._currentViewElement = newViewElement;
    } else {
      // If no active view is found, ensure the container is empty
      container.innerHTML = ''; // Clear any existing content
    }
  }

  // Handler for the 'team-selected' event from TeamSelectorView
  handleTeamSelected(event) {
    if (this.sessionManager) {
      this.sessionManager._handleTeamSelected(event);
    }
    console.log('Team selected:', event.detail);
    
    // After a team is selected, we should transition to the waiting view.
    this.switchView('waiting-view');
  }

  handleTeamCleared() {
    console.log('Team cleared event received in ViewManager, switching to team-selector.');
    this.switchView('team-selector');
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

    // Determine the initial view based on localStorage
    const storedTeamId = localStorage.getItem('teamId');
    if (storedTeamId) {
      // If teamId is set, go to the waiting view
      this.activeViewName = 'waiting-view';
    } else {
      // If teamId is not set, start with the team selector
      this.activeViewName = 'team-selector';
    }

    this._checkDeviceRegistration();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this._registrationTimeout) {
      clearTimeout(this._registrationTimeout);
    }
    if (this._registrationSubscription) {
      this._registrationSubscription.unsubscribe();
    }
  }

  _checkDeviceRegistration() {
    if (!this.deviceId) {
      return;
    }

    const terminalId = `terminal-${this.deviceId}`;
    console.log(`Checking registration for device: ${terminalId}`);

    // Wait for the observer to be initialized
    const checkRegistration = () => {
      if (!this.homieObserver) {
        setTimeout(checkRegistration, 100);
        return;
      }

      // Let's use a timeout to decide if the device is missing.
      this._registrationTimeout = setTimeout(() => {
        console.warn(`Device ${terminalId} not found in MQTT broker after timeout. Resetting local storage.`);
        this._resetApp();
      }, 5000); // 5 seconds timeout to find the device

      this._registrationSubscription = this.homieObserver.created$.subscribe(event => {
        if (event.device && event.device.id === terminalId) {
          console.log(`Device ${terminalId} confirmed registered in MQTT.`);
          clearTimeout(this._registrationTimeout);
          this._registrationSubscription.unsubscribe();
          this._registrationSubscription = null;
        }
      });
    };

    checkRegistration();
  }

  _resetApp() {
    console.log('Resetting local app storage from ViewManager...');
    localStorage.removeItem('teamId');
    localStorage.removeItem('teamName');
    
    if (this.sessionManager) {
      this.sessionManager.dispatchEvent(new CustomEvent('team-cleared', {
        bubbles: true,
        composed: true
      }));
    }
    this.handleTeamCleared();
  }

  static styles = css`
    .view-manager {
      width: 100%;
      height: 100%;
      display: flex;
      justify-content: center;
      align-items: center;
      box-sizing: border-box;
    }
  `;
}

customElements.define('view-manager', ViewManager);

export { ViewManager };
