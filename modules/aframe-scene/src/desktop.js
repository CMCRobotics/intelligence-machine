import AFRAME from 'aframe';
import { html, render } from 'lit-html';

import 'js-yaml';
import 'loglevel';

import 'aframe-extras';
import 'aframe-environment-component';

import './components/load-fragment.js';
import './components/linear-animation.js';
import './components/ar-utils.js';
import './components/floating-in-jar.js';
import './components/timed-sound.js';
import './components/homie-brain-scale.js';

import { createMqttHomieObserver } from '@cmcrobotics/homie-lit';

// Import Pronolab components
import { Session } from './pronolab/core/session.ts';
import { ViewManager } from './pronolab/view/view-manager.ts';
import { ImageView } from './pronolab/view/image-view.ts';
import { AudioView } from './pronolab/view/audio-view.ts';
import { PoseView } from './pronolab/view/pose-view.ts';

// Import HUD functions
import { HudPanel } from './lit-templates/hud-panel.js';

// Import the asset template function
import { renderAssets } from './lit-templates/assets.js';
import { renderSceneLab } from './lit-templates/scene-lab.js';

// Polyfill global Buffer
import { Buffer } from 'buffer';
window.Buffer = Buffer;

// Define the lit-html template function
const renderLabScene = (options = {}) => {
    const { showShadows = false } = options; // Default to true if not specified

    // Construct the environment attributes string, conditionally including shadow
    const environmentAttributes = `preset: forest; dressing: trees; dressingAmount: 100; dressingColor: #ceebd5; dressingScale: 15; fog: 0.6; fogColor: #adc2d6; playArea: 30; ${showShadows ? 'shadow: true;' : 'shadow: false;'} horizonsColor: #b1d1f0; skyColor: #88c0f4; stageSize: 200; lighting: none; `;

    return html`
        <a-scene id="aframe-scene" light="defaultLightsEnabled: false"
                 sound="src: #forestAmbientSound; loop: true; volume: 0.10; autoplay: true">
            ${renderAssets()} 

            <a-entity environment="${environmentAttributes}"></a-entity>
            
            ${renderSceneLab()} 
        
            <!-- Explicitly define camera and disable controls -->
            <a-entity camera look-controls="enabled: false;" position="0 1.6 0"></a-entity>
        </a-scene>

        <!-- <hud-panel></hud-panel> -->
    `;
};

// Find the container and render the scene directly
const container = document.getElementById('aframe-container');
if (container) {
    // Initialize Pronolab components
    const session = new Session();
    const viewManager = new ViewManager(container, session);

    // Add views to the ViewManager
    viewManager.addView('image-view', new ImageView(container, session));
    viewManager.addView('audio-view', new AudioView(container, session));
    viewManager.addView('pose-view', new PoseView(container, session));
    // Add other views if they exist and are needed

    // Initialize the ViewManager
    viewManager.init();

    // Set a default device ID for testing purposes if not already set
    if (!localStorage.getItem('deviceId')) {
        localStorage.setItem('deviceId', 'test-device-123');
    }

    // Optionally, set an initial view or trigger a model load
    // For example, to show the image view by default:
    // viewManager.setActiveView('image-view'); // This would need to be called after init() or handled by a message

    // Default options: shadows enabled
    const options = { showShadows: true };
    render(renderLabScene(options), container);

    const observer = createMqttHomieObserver('ws://localhost:9001');

    observer.updated$.subscribe(
        (event) => {
            if (event.type == 'property') {
                if (event.node.id === 'brain' && event.property.id === 'scale') {
                    const brainEl = document.getElementById('brain');
                    if (brainEl) {
                        const scale = parseFloat(event.property.value);
                        if (!isNaN(scale)) {
                            brainEl.setAttribute('scale', { x: scale, y: scale, z: scale });
                        }
                    }
                }
            }
        },
        (error) => {
            console.error('Error in subscription:', error);
        }
    );

    observer.subscribe('team-white/brain/+');


    // showHudPanel(); // Show the HUD panel when the scene is rendered

    // Example of how you might toggle shadows later (e.g., via a UI element)
    // setTimeout(() => {
    //     options.showShadows = false;
    //     render(renderLabScene(options), container);
    //     console.log("Shadows turned off");
    // }, 5000);
} else {
    console.error("Could not find #aframe-container to render the scene.");
}
