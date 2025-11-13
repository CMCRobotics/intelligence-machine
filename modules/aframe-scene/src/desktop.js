import AFRAME from 'aframe';
import { LitElement, html, css } from 'lit'; // Import LitElement, html, css for ViewManager
import { render } from 'lit-html';

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

// Import HUD functions
import { HudPanel } from './lit-templates/hud-panel.js';

// Import the asset template function
import { renderAssets } from './lit-templates/assets.js';
import { renderSceneLab } from './lit-templates/scene-lab.js';

// Import ViewManager
import { ViewManager } from './lit-templates/ViewManager.js';


// Polyfill global Buffer
import { Buffer } from 'buffer';
window.Buffer = Buffer;

// Define the lit-html template function
const renderLabScene = (options = {}) => {
    const { showShadows = false, modelURL, metadataURL } = options; // Default to true if not specified

    // Construct the environment attributes string, conditionally including shadow
    const environmentAttributes = `preset: forest; dressing: trees; dressingAmount: 100; dressingColor: #ceebd5; dressingScale: 15; fog: 0.6; fogColor: #adc2d6; playArea: 30; ${showShadows ? 'shadow: true;' : 'shadow: false;'} horizonsColor: #b1d1f0; skyColor: #88c0f4; stageSize: 200; lighting: none; `;

    return html`
        <a-scene id="aframe-scene" light="defaultLightsEnabled: false"
                 sound="src: #forestAmbientSound; loop: true; volume: 0.10; autoplay: true">
            ${renderAssets()}

            <a-entity environment="${environmentAttributes}"></a-entity>

            ${renderSceneLab()}

            <!-- Explicitly define camera and disable controls -->
            <a-entity camera="fov: 50; zoom: 0.7" look-controls="enabled: false" position="-2.43017 1.6 0.83541" rotation="0 -50.80072994747931 0"></a-entity>
        </a-scene>

        <hud-panel .modelURL=${modelURL} .metadataURL=${metadataURL}></hud-panel>
    `;
};

// Find the container and render the scene directly
const container = document.getElementById('aframe-container');
if (container) {

    // Default options: shadows enabled
    const options = { 
        showShadows: true,
        modelURL: 'models/sign-language/model.json',
        metadataURL: 'models/sign-language/metadata.json'
    };
    render(renderLabScene(options), container);

    // const observer = createMqttHomieObserver('ws://localhost:9001');

    // observer.updated$.subscribe(
    //     (event) => {
    //         if (event.type == 'property') {
    //             if (event.node.id === 'brain' && event.property.id === 'scale') {
    //                 const brainEl = document.getElementById('brain');
    //                 if (brainEl) {
    //                     const scale = parseFloat(event.property.value);
    //                     if (!isNaN(scale)) {
    //                         brainEl.setAttribute('scale', { x: scale, y: scale, z: scale });
    //                     }
    //                 }
    //             }
    //         }
    //     },
    //     (error) => {
    //         console.error('Error in subscription:', error);
    //     }
    // );

    // observer.subscribe('team-white/brain/+');

} else {
    console.error("Could not find #aframe-container to render the scene.");
}
