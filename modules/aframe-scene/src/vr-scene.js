import AFRAME from 'aframe';
import { html, render } from 'lit-html';

import 'js-yaml';
import 'loglevel';

import 'aframe-orbit-controls';
import 'aframe-extras';
import 'aframe-environment-component';

import './components/load-fragment.js';
import './components/linear-animation.js';
import './components/ar-utils.js';
import './components/floating-in-jar.js';
import './components/timed-sound.js';

// Import HUD functions
import { showHudPanel, hideHudPanel } from './hud-panel.js';

// Polyfill global Buffer
import { Buffer } from 'buffer';
window.Buffer = Buffer;

// Define the lit-html template function
const renderLabScene = (options = {}) => {
    const { showShadows = true } = options; // Default to true if not specified

    // Construct the environment attributes string, conditionally including shadow
    const environmentAttributes = `preset: forest; dressing: trees; dressingAmount: 100; dressingColor: #ceebd5; dressingScale: 15; fog: 0.6; fogColor: #adc2d6; playArea: 30; ${showShadows ? 'shadow: true;' : 'shadow: false;'} horizonsColor: #b1d1f0; skyColor: #88c0f4; stageSize: 200; lighting: none; `;

    return html`
        <a-scene id="aframe-scene" light="defaultLightsEnabled: false"
                 sound="src: #forestAmbientSound; loop: true; volume: 0.10; autoplay: true">
            <a-assets>
                <!-- https://cmc-cdn.web.cern.ch/assets/intelligence-machine/ -->
                <a-asset-item id="floor"              src="assets/floor.glb"  ></a-asset-item>
                <a-asset-item id="base"               src="assets/base.glb"  ></a-asset-item>
                <a-asset-item id="dome"               src="assets/dome.glb"  ></a-asset-item>
                <a-asset-item id="cyberbrain"         src="assets/cyberbrain.glb"  ></a-asset-item>
                <a-asset-item id="lab-device"         src="assets/lab-device.glb"  ></a-asset-item>
                <a-asset-item id="lab-device-2"       src="assets/lab-device-2.glb"  ></a-asset-item>
                <a-asset-item id="lab-chair"          src="assets/lab-chair.glb"  ></a-asset-item>
                <a-asset-item id="lab-desk"           src="assets/lab-desk.glb"  ></a-asset-item>
                <a-asset-item id="lab-computer"       src="assets/lab-computer.glb"  ></a-asset-item>
                
                <!-- New Sound Assets -->
                <a-asset-item id="forestAmbientSound" src="assets/forest-ambient.mp3" response-type="arraybuffer"></a-asset-item>
                <a-asset-item id="underwaterSound"    src="assets/underwater-ambience.mp3" response-type="arraybuffer"></a-asset-item>
                <a-asset-item id="voltmeterSound"     src="assets/voltmeter.mp3" response-type="arraybuffer"></a-asset-item>
                <a-asset-item id="computerSound"      src="assets/computer.mp3" response-type="arraybuffer"></a-asset-item>
            </a-assets>

            <a-entity environment="${environmentAttributes}"></a-entity>
            <a-entity light="intensity: 0.4; type: ambient" position="-1.17106 1.00163 2.32528"></a-entity>
            <a-entity light="decay: -2.96; intensity: 0.04; type: spot; castShadow: true; shadowCameraFov: 75.97; shadowCameraNear: 0.78; shadowCameraBottom: -4.24; shadowMapWidth: 512.27" position="-85.94564 27.28891 10.22902" rotation="0 -44.67065449737463 0"></a-entity>
            
            <a-entity load-fragment="src: /scene-lab.html; templateId: scene-lab"></a-entity>
        
            <!-- Default A-Frame Camera (no explicit camera entity needed for default VR POV) -->
        </a-scene>
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
