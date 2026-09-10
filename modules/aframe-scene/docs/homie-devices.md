# MQTT Homie Devices Inventory

This document inventories all MQTT Homie devices, nodes, and properties referenced throughout the project. The codebase leverages the Homie convention to coordinate states, views, uploading/testing of Teachable Machine models, and team scores between interactive terminals and the visual simulation.

---

## 1. Device Hierarchy Overview

The architecture divides entities into two primary device classes, which communicate over MQTT (typically hosted on port `9001` via WebSockets):

1. **Terminal Devices (`terminal-<deviceId>`)**: Individual interactive user interface terminals.
2. **Team Devices (`team-<teamId>`)**: Represents team identities, states, scores, and uploaded models.

---

## 2. Detailed Inventory

### A. Terminal Devices (`terminal-<deviceId>`)

These represent terminal screens. A terminal device's ID is dynamically generated as a short UUID and stored in the terminal's `localStorage` as `deviceId`.

| Node | Property | Direction | Payload Type | Description | References |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`info`** | `team` | Publish / Subscribe | `String` | Associated team ID for this terminal (e.g. `"team-white"`, `"team-red"`). | `SessionManager.js`, `ScoreManager.js`, `TerminalView.js` |
| **`ui-control`** | `switch` | Publish / Subscribe | `String` | Commands terminal to switch views. Active views: `"team-selector"`, `"waiting-view"`, `"teachable-machine-image"`, `"teachable-machine-upload"`. | `TerminalViewSwitcher.js`, `ViewManager.js`, `TerminalView.js` |
| **`model-upload`** | `name` | Publish / Subscribe | `String` | Name of the Teachable Machine model selected for uploading (e.g. `"greeter"`). | `TerminalViewSwitcher.js`, `TeachableMachineUploadView.js` |
| **`model-upload`** | `type` | Publish / Subscribe | `String` | Modality type of the model being uploaded (e.g. `"image"`, `"pose"`, `"speech"`). | `TerminalViewSwitcher.js`, `TeachableMachineUploadView.js` |
| **`activeModel`** | `set` | Publish / Subscribe | `JSON String` | Sets the current active model parameters for testing. <br> *Payload Schema:* `{ "name": String, "uploaderTeamId": String, "type": String }` | `TerminalViewSwitcher.js`, `TeachableMachineImageView.js` |
| **`activeModel`** | `test` | Publish / Subscribe | `JSON String` | Triggers a validation test block. <br> *Payload Schema:* `{ "confidence": Number, "duration": Number, "class": Number, "overallTimeout": Number }` | `TerminalViewSwitcher.js`, `TeachableMachineImageView.js` |
| **`activeModel`** | `testSuccess` | Publish | `JSON String` | Published by terminal when a test succeeds. Increases team score. <br> *Payload Schema:* `{ "confidence": Number, "duration": Number, "class": Number, "overallTimeout": Number, "timestamp": Number }` | `TeachableMachineImageView.js`, `ScoreManager.js` |
| **`activeModel`** | `test-result` | Publish | `JSON String` | Published by terminal when test fails or times out. <br> *Payload Schema:* `{ "success": false }` | `TeachableMachineImageView.js` |

### B. Team Devices (`team-<teamId>`)

These represent logical team structures (e.g., `"team-white"`, `"team-red"`).

| Node | Property | Direction | Payload Type | Description | References |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`info`** | `name` | Publish / Subscribe | `String` | The human-readable name of the team, which also matches the CSS color (e.g., `"White"`, `"Red"`). | `SessionManager.js`, `ScoreManager.js`, `TeamView.js`, `TerminalView.js` |
| **`info`** | `score` | Publish / Subscribe | `Integer` | The cumulative score of the team. Incremented dynamically upon `testSuccess` signals. | `ScoreManager.js`, `TeamView.js`, `team-score-manager.js` |
| **`model-<modelName>`** *(Dynamic)* | `modelName` | Publish / Subscribe | `String` | Name of the uploaded model. | `server.js`, `TeamView.js`, `TerminalViewSwitcher.js` |
| **`model-<modelName>`** *(Dynamic)* | `type` | Publish / Subscribe | `String` | Model type (e.g. `"image"`, `"pose"`, `"speech"`). | `server.js`, `TeamView.js` |
| **`model-<modelName>`** *(Dynamic)* | `modelUrl` | Publish / Subscribe | `String` | Relative URL to the static folder holding model parameters. | `server.js` |
| **`model-<modelName>`** *(Dynamic)* | `metadataUrl` | Publish / Subscribe | `String` | Relative URL to the model's metadata definition. | `server.js` |
| **`model-<modelName>`** *(Dynamic)* | `timestamp` | Publish / Subscribe | `String` | ISO timestamp of the model's publication. | `server.js`, `TeamView.js` |
| **`model-<modelName>`** *(Dynamic)* | `terminalId` | Publish / Subscribe | `String` | The unique ID of the terminal that uploaded this model. | `server.js`, `TeamView.js`, `TerminalViewSwitcher.js` |
| **`model-<modelName>`** *(Dynamic)* | `classCount` | Publish / Subscribe | `Integer` | Number of distinct labels in the model. | `server.js` |
| **`model-<modelName>`** *(Dynamic)* | `storagePath` | Publish / Subscribe | `String` | Folder location on disk. | `server.js` |
| **`brain`** | `scale` | Subscribe | `Float` | Directly updates the 3D visual scale of the brain entity (reflects current score). | `desktop.js` |

