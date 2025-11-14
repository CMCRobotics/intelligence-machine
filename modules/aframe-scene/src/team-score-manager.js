import { createMqttHomieObserver } from '@cmcrobotics/homie-lit';
import { filter } from 'rxjs/operators';
import { merge } from 'rxjs';

let previousScore = 0;

function updateBrainScale(score) {
    const brainEntity = document.getElementById('brain');
    if (brainEntity) {
        const minScore = 0;
        const maxScore = 30;
        const minScale = 0.4;
        const maxScale = 1.5;

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
                    floatingComponent.data.bobbingSpeed = 1.0;
                    floatingComponent.data.rockingSpeed = 0.5;
                }, 4000);
            }

            // Add halo effect
            const scene = document.querySelector('a-scene');
            if (scene) {
                const existingHalo = document.getElementById('brain-halo');
                if (existingHalo) {
                    existingHalo.parentNode.removeChild(existingHalo);
                }

                const halo = document.createElement('a-sphere');
                halo.setAttribute('id', 'brain-halo');
                const brainPosition = brainEntity.getAttribute('position');
                halo.setAttribute('position', brainPosition);
                const haloRadius = clampedScale * 1.2;
                halo.setAttribute('radius', haloRadius);
                halo.setAttribute('material', 'color: yellow; opacity: 0.4; transparent: true; side: back');
                
                scene.appendChild(halo);

                setTimeout(() => {
                    const haloToRemove = document.getElementById('brain-halo');
                    if (haloToRemove) {
                        haloToRemove.parentNode.removeChild(haloToRemove);
                    }
                }, 4000);
            }
        }
        previousScore = score;
    }
}

export function initializeTeamScoreManager() {
    const teamId = localStorage.getItem('teamId');
    if (teamId) {
        try {
            const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
            const homieObserver = createMqttHomieObserver(`${scheme}://${window.location.hostname}:9001`);
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

    // Add team-colored circle around the base
    const teamColorName = localStorage.getItem('teamName');
    if (teamColorName) {
        const scene = document.querySelector('a-scene');
        if (scene) {
                // Add team-colored circle around the base
                const teamColorName = localStorage.getItem('teamName');
                if (teamColorName) {
                    const scene = document.querySelector('a-scene');
                    if (scene) {
                        // Wait for the scene to load before adding the circle
                        scene.addEventListener('loaded', function () {
                            const ring = document.createElement('a-ring');
                            ring.setAttribute('position', `-1 0.15 -4.73`);
                            ring.setAttribute('rotation', '-90 0 0');
                            ring.setAttribute('radius-outer', '2.8');
                            ring.setAttribute('radius-inner', '2.4');
                            ring.setAttribute('color', teamColorName);
                            ring.setAttribute('material', 'shader: flat');
                            scene.appendChild(ring);
                        });
                    }
                }


        }
    }
}
