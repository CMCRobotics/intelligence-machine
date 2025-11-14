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


// Import the asset template function
import { renderAssets } from './lit-templates/assets.js';
import { renderSceneLab } from './lit-templates/scene-lab.js';
import { createMqttHomieObserver } from '@cmcrobotics/homie-lit';
import { filter } from 'rxjs/operators';
import { merge } from 'rxjs';

// Polyfill global Buffer
import { Buffer } from 'buffer';
window.Buffer = Buffer;

// Parse URL parameters to set team-id in local storage
const urlParams = new URLSearchParams(window.location.search);
const teamColor = urlParams.get('team');
if (teamColor) {
  const lowerCaseTeamColor = teamColor.toLowerCase();
  const teamId = `team-${lowerCaseTeamColor}`;
  localStorage.setItem('teamId', teamId);
  localStorage.setItem('teamName', lowerCaseTeamColor);
  console.log(`Team ID set to: ${teamId}`);
  console.log(`Team Name set to: ${lowerCaseTeamColor}`);

  try {
    const homieObserver = createMqttHomieObserver("ws://localhost:9001");
    const scoreTopic = `${teamId}/info/score`;

    homieObserver.subscribe('+/info/score');

    const scoreUpdates$ = merge(
        homieObserver.created$,
        homieObserver.updated$
    ).pipe(
        filter(event => event.type === 'property' && event.device.id === teamId && event.node.id === 'info' && event.property.id === 'score')
    );

    scoreUpdates$.subscribe(event => {
        const score = parseInt(event.property.value, 10);
        if (!isNaN(score)) {
            console.log(`Score for team ${teamId} is now: ${score}`);
            updateBrainScale(score);
        }
    });

    console.log(`Subscribed to score updates for team ${teamId}`);
  } catch (error) {
    console.error('Error initializing Homie observer:', error);
  }
}

let previousScore = 0;

// Define the lit-html template function
const renderLabScene = (options = {}) => {
    const { showShadows = true } = options; // Default to true if not specified

    // Construct the environment attributes string, conditionally including shadow
    const environmentAttributes = `preset: forest; dressing: trees; dressingAmount: 100; dressingColor: #ceebd5; dressingScale: 15; fog: 0.6; fogColor: #adc2d6; playArea: 30; ${showShadows ? 'shadow: true;' : 'shadow: false;'} horizonsColor: #b1d1f0; skyColor: #88c0f4; stageSize: 200; lighting: none; `;

    return html`
        <a-scene id="aframe-scene" light="defaultLightsEnabled: false"
                 sound="src: #forestAmbientSound; loop: true; volume: 0.10; autoplay: true">
            ${renderAssets()} 

            <a-entity environment="${environmentAttributes}"></a-entity>
            
            ${renderSceneLab()} 
        
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

function updateBrainScale(score) {
    const brainEntity = document.getElementById('brain');
    if (brainEntity) {
        const minScore = 0;
        const maxScore = 30;
        const minScale = 0.4;
        const maxScale = 1.6;

        const scale = minScale + (score - minScore) / (maxScore - minScore) * (maxScale - minScale);
        const clampedScale = Math.max(minScale, Math.min(maxScale, scale));

        brainEntity.setAttribute('scale', `${clampedScale} ${clampedScale} ${clampedScale}`);

        if (score > previousScore) {
            const floatingComponent = brainEntity.components['floating-in-jar'];
            if (floatingComponent) {
                const originalBobbingSpeed = floatingComponent.data.bobbingSpeed;
                const originalRockingSpeed = floatingComponent.data.rockingSpeed;

                floatingComponent.data.bobbingSpeed *= 10;
                floatingComponent.data.rockingSpeed *= 10;

                setTimeout(() => {
                    floatingComponent.data.bobbingSpeed = originalBobbingSpeed;
                    floatingComponent.data.rockingSpeed = originalRockingSpeed;
                }, 5000);
            }
        }
        previousScore = score;
    }
}
