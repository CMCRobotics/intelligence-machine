import { html } from 'lit-html';

// Define the lit-html template function for assets
export const renderAssets = () => html`
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
`;
