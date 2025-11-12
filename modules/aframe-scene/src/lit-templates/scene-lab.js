import { html } from 'lit-html';

// Define the lit-html template function for assets
export const renderSceneLab = () => html`
    <a-entity id="floor" gltf-model="#floor" position="0 0.1 -10" scale="1.5 1.5 1.5" material="opacity: 0.1;"></a-entity>

    <!-- Base with underwater sound -->
    <a-entity id="base" gltf-model="#base" scale="1.2 1.2 1.2" rotation="0 180 0" position="-2.204 0.3 -5.697"
            sound="src: #underwaterSound; loop: true; volume: 0.9; autoplay: true; positional: true"></a-entity>

    <a-entity id="brain" gltf-model="#cyberbrain" position="-0.87447 2.54043 -4.50151" rotation="0 -30 0" floating-in-jar>
       <a-entity geometry="primitive: sphere; radius: 0.06" scale="1 0.9 1" material="color: #091f27" position="-0.20225 0.09337 0.95721"></a-entity>
       <a-entity geometry="primitive: sphere; radius: 0.06" scale="1 0.9 1" material="color: #091f27" position="0.20224 0.10982 0.96035"></a-entity>
        <!-- a-entity geometry="primitive: torus; arc: -120.01; radius: 0.23; radiusTubular: 0.009; segmentsRadial: 34; segmentsTubular: 34" material="color: #000000" 0.03123="" 0.00383="" 0.97574"="" rotation="21.772396214971284 1.1682609442717484 -25.956134035016554" position="-0.01916 0.05287 1.00781"></a-entity -->
    </a-entity>

    <a-entity light="color: #5cceff; decay: 4.54; distance: 3.99; intensity: 12.994; penumbra: 1; type: spot; target: #brain; castShadow: true; shadowRadius: -0.02" position="-0.5 1.26488 -3.75003" data-aframe-default-light="" aframe-injected=""></a-entity>

    <a-sphere id="tank" color="blue" radius="3" position="-0.85857 2.41091 -4.51556" material="opacity: 0.3; color: #8a8aff" geometry="radius: 1.9"></a-sphere>
    <a-entity id="dome" gltf-model="#dome" scale="1.5 1.5 1.5" position="0.78007 4.42478 -3.2805"></a-entity>

    <a-entity id="lab-desk1" gltf-model="#lab-desk" position="-0.44104 0.2 -20.7471" scale="3.5 3.5 3.5" rotation="0 -95.16485202445895 0"></a-entity>

    <a-entity id="lab-desk2" gltf-model="#lab-desk" position="-14.49316 0.2 -14.49689" scale="3.5 3.5 3.5" rotation="0 -35.423688641758275 0"></a-entity>

    <a-entity id="lab-chair" gltf-model="#lab-chair" position="-13.25501 0.2 -12.16489" scale="3 3 3" rotation="0 -56.79787918911364 0"></a-entity>

    <a-entity id="lab-chair2" gltf-model="#lab-chair" position="-0.64515 0.2 -18.93082" scale="3 3 3" rotation="0 -64.84965508408709 0"></a-entity>

    <!-- Lab Computer with timed computer sound -->
    <a-entity id="lab-computer" gltf-model="#lab-computer" position="-0.86796 3.51759 -20.19095" scale="0.8 0.8 0.8" rotation="0 179.01550646846954 0"
            sound="src: #computerSound; volume: 0.2; positional: true; autoplay: false"
            timed-sound="src: #computerSound; interval: 90000; volume: 0.2; start-offset: 20000"></a-entity>

    <!-- Lab Device 1 with timed voltmeter sound -->
    <a-entity id="lab-device-1" gltf-model="#lab-device-2"   position="-3.787 -0.25 1.092" scale="0.5 0.5 0.5" rotation="0 -28 0"
            sound="src: #voltmeterSound; volume: 0.2; positional: true; autoplay: false"
            timed-sound="src: #voltmeterSound; interval: 30000; volume: 0.2; start-offset: 10000"></a-entity>

    <a-entity id="lab-device-2" gltf-model="#lab-device"     position="2.277 -0.3 -1.54"  scale="0.5 0.5 0.5" rotation="0 -132 0"></a-entity>
    <a-entity id="lab-device-3"   gltf-model="#lab-device-2" position="2.49951 -0.62316 -10.17007" scale="0.6 0.8 0.6" rotation="0 -25.136231430184345 0"></a-entity>

`;
