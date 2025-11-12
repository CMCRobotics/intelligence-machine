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

        <hud-panel></hud-panel>
    `;
};

// Find the container and render the scene directly
const container = document.getElementById('aframe-container');
if (container) {
    // Default options: shadows enabled
    const options = { showShadows: true };
    render(renderLabScene(options), container);

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
