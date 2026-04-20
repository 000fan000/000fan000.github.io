const BOARD_SIZE = { width: 2200, height: 1400 };
const STORAGE_KEY = "prompt-workflow-lab-v5";
const INITIAL_WORKFLOW_NAME = "Reusable prompt workflow";

function uid(prefix) {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function extractVariables(template) {
  const matches = template.matchAll(/{{\s*([a-zA-Z0-9_.-]+)\s*}}/g);
  return Array.from(new Set(Array.from(matches, (match) => match[1])));
}

function createModelPreset(overrides = {}) {
  return {
    id: uid("model"),
    label: "Fast Draft",
    name: "gpt-4.1-mini",
    reasoningMode: "off",
    temperature: 0.4,
    maxTokens: 700,
    ...overrides
  };
}

function createEndpointConfig(overrides = {}) {
  const models = Array.isArray(overrides.models) && overrides.models.length
    ? overrides.models.map((model, index) => normalizeModelPreset(model, index))
    : [createModelPreset()];

  return {
    id: uid("endpoint"),
    label: "OpenAI",
    provider: "openai",
    apiKey: "",
    defaultModelPresetId: models[0]?.id || "",
    models,
    ...overrides,
    models
  };
}

function normalizeModelPreset(model, index = 0) {
  return createModelPreset({
    ...model,
    id: model?.id || `model_${index + 1}`,
    label: model?.label || model?.title || `Model ${index + 1}`,
    name: model?.name || model?.model || "gpt-4.1-mini",
    reasoningMode: model?.reasoningMode || model?.defaults?.reasoningMode || "off",
    temperature: Number(model?.temperature ?? model?.defaults?.temperature ?? 0.4),
    maxTokens: Number(model?.maxTokens ?? model?.defaults?.maxTokens ?? 700)
  });
}

function normalizeEndpointConfig(endpoint, index = 0) {
  const normalizedModels = Array.isArray(endpoint?.models) && endpoint.models.length
    ? endpoint.models.map((model, modelIndex) => normalizeModelPreset(model, modelIndex))
    : [
        normalizeModelPreset({
          label: endpoint?.defaults?.model ? "Imported default" : "Fast Draft",
          name: endpoint?.defaults?.model || endpoint?.defaultModel || endpoint?.model || "gpt-4.1-mini",
          reasoningMode: endpoint?.defaults?.reasoningMode || endpoint?.defaultReasoningMode || endpoint?.reasoningMode || "off",
          temperature: Number(endpoint?.defaults?.temperature ?? endpoint?.defaultTemperature ?? endpoint?.temperature ?? 0.4),
          maxTokens: Number(endpoint?.defaults?.maxTokens ?? endpoint?.defaultMaxTokens ?? endpoint?.maxTokens ?? 700)
        }, 0)
      ];

  const defaultModelPresetId = normalizedModels.some((model) => model.id === endpoint?.defaultModelPresetId)
    ? endpoint.defaultModelPresetId
    : normalizedModels[0].id;

  return createEndpointConfig({
    ...endpoint,
    id: endpoint?.id || `endpoint_${index + 1}`,
    label: endpoint?.label || endpoint?.name || `Provider ${index + 1}`,
    provider: endpoint?.provider || "openai",
    apiKey: endpoint?.apiKey || "",
    defaultModelPresetId,
    models: normalizedModels
  });
}

function getDefaultEndpointConfigs() {
  return [
    normalizeEndpointConfig({
      id: "endpoint_openai_primary",
      label: "OpenAI",
      provider: "openai",
      defaultModelPresetId: "model_fast_draft",
      models: [
        {
          id: "model_fast_draft",
          label: "Fast Draft",
          name: "gpt-4.1-mini",
          reasoningMode: "off",
          temperature: 0.6,
          maxTokens: 700
        },
        {
          id: "model_reasoner",
          label: "Reasoner",
          name: "gpt-4.1",
          reasoningMode: "on",
          temperature: 0.3,
          maxTokens: 1200
        }
      ]
    })
  ];
}

const nodeBlueprints = {
  input: () => ({
    title: "Input Variable",
    key: `input_${Math.random().toString(36).slice(2, 5)}`,
    defaultValue: ""
  }),
  step: () => ({
    title: "Prompt Step",
    endpointId: "",
    modelPresetId: "",
    prompt: "Draft a concise answer about {{topic}} for {{audience}}.",
    provider: "openai",
    model: "gpt-4.1-mini",
    reasoningMode: "off",
    temperature: 0.4,
    maxTokens: 700,
    outputKey: `step_${Math.random().toString(36).slice(2, 6)}`,
    recursionEnabled: false,
    maxIterations: 3,
    stopCondition: "Stop when the answer is clear, specific, and no major flaws remain.",
    recursionPrompt: "Review the previous answer, fix weaknesses, then produce a stronger version."
  }),
  output: () => ({
    title: "Output Target",
    key: "final_result",
    format: "markdown",
    schemaHint: "headline, bullets, next_steps"
  })
};

const state = {
  workflowName: INITIAL_WORKFLOW_NAME,
  nodes: [],
  edges: [],
  endpointConfigs: [],
  selectedNodeId: null,
  linkDraft: null,
  dragState: null,
  sidebarOpen: false,
  configOpen: false
};

const elements = {
  leftSidebar: document.getElementById("leftSidebar"),
  closeSidebarButton: document.getElementById("closeSidebarButton"),
  toggleSidebarButton: document.getElementById("toggleSidebarButton"),
  openConfigButton: document.getElementById("openConfigButton"),
  closeConfigButton: document.getElementById("closeConfigButton"),
  configOverlay: document.getElementById("configOverlay"),
  endpointConfigList: document.getElementById("endpointConfigList"),
  addEndpointButton: document.getElementById("addEndpointButton"),
  nodesLayer: document.getElementById("nodesLayer"),
  connectionsLayer: document.getElementById("connectionsLayer"),
  boardSurface: document.getElementById("boardSurface"),
  boardViewport: document.getElementById("boardViewport"),
  workflowTitle: document.getElementById("workflowTitle"),
  graphStatus: document.getElementById("graphStatus"),
  graphMeta: document.getElementById("graphMeta"),
  workflowNameInput: document.getElementById("workflowNameInput"),
  importJsonInput: document.getElementById("importJsonInput")
};

function getSampleState() {
  return {
    workflowName: "Recursive launch copy workflow",
    endpointConfigs: getDefaultEndpointConfigs(),
    selectedNodeId: null,
    linkDraft: null,
    dragState: null,
    nodes: [
      {
        id: "input_topic",
        type: "input",
        x: 110,
        y: 220,
        data: {
          title: "Topic",
          key: "topic",
          defaultValue: "AI notebook for product teams"
        }
      },
      {
        id: "input_audience",
        type: "input",
        x: 110,
        y: 430,
        data: {
          title: "Audience",
          key: "audience",
          defaultValue: "technical founders"
        }
      },
      {
        id: "step_draft",
        type: "step",
        x: 500,
        y: 210,
        data: {
          title: "Draft core message",
          endpointId: "endpoint_openai_primary",
          modelPresetId: "model_fast_draft",
          prompt: "Write launch copy about {{topic}} for {{audience}}. Give me a sharp hook, three value points, and a CTA.",
          provider: "openai",
          model: "gpt-4.1-mini",
          reasoningMode: "off",
          temperature: 0.7,
          maxTokens: 700,
          outputKey: "draft_copy",
          recursionEnabled: false,
          maxIterations: 1,
          stopCondition: "",
          recursionPrompt: ""
        }
      },
      {
        id: "step_refine",
        type: "step",
        x: 980,
        y: 180,
        data: {
          title: "Refine recursively",
          endpointId: "endpoint_openai_primary",
          modelPresetId: "model_reasoner",
          prompt: "Improve the upstream draft for {{audience}}. Make the argument tighter, remove filler, and sharpen the CTA.",
          provider: "openai",
          model: "gpt-4.1",
          reasoningMode: "on",
          temperature: 0.3,
          maxTokens: 900,
          outputKey: "refined_copy",
          recursionEnabled: true,
          maxIterations: 3,
          stopCondition: "Stop when the copy feels differentiated, concise, and convincing.",
          recursionPrompt: "Critique the previous version, list the weakest parts, then rewrite a stronger version."
        }
      },
      {
        id: "output_final",
        type: "output",
        x: 1480,
        y: 260,
        data: {
          title: "Final marketing copy",
          key: "launch_copy",
          format: "markdown",
          schemaHint: "hook, bullets[], cta"
        }
      }
    ],
    edges: [
      { id: uid("edge"), from: "input_topic", to: "step_draft" },
      { id: uid("edge"), from: "input_audience", to: "step_draft" },
      { id: uid("edge"), from: "input_audience", to: "step_refine" },
      { id: uid("edge"), from: "step_draft", to: "step_refine" },
      { id: uid("edge"), from: "step_refine", to: "output_final" }
    ]
  };
}

function getEndpointConfig(endpointId) {
  return state.endpointConfigs.find((endpoint) => endpoint.id === endpointId) || null;
}

function getModelPreset(endpointId, modelPresetId) {
  const endpoint = getEndpointConfig(endpointId);
  if (!endpoint) {
    return null;
  }
  return endpoint.models.find((model) => model.id === modelPresetId) || null;
}

function persistState() {
  const snapshot = {
    workflowName: state.workflowName,
    nodes: state.nodes,
    edges: state.edges,
    endpointConfigs: state.endpointConfigs
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
}

function ensureStepConfigReferences() {
  const defaultEndpoint = state.endpointConfigs[0] || null;

  state.nodes.forEach((node) => {
    if (node.type !== "step") {
      return;
    }

    const endpoint = getEndpointConfig(node.data.endpointId) || defaultEndpoint;
    if (!endpoint) {
      node.data.endpointId = "";
      node.data.modelPresetId = "";
      return;
    }

    node.data.endpointId = endpoint.id;
    const modelPreset = getModelPreset(endpoint.id, node.data.modelPresetId)
      || getModelPreset(endpoint.id, endpoint.defaultModelPresetId)
      || endpoint.models[0];

    if (modelPreset) {
      node.data.modelPresetId = modelPreset.id;
      if (!node.data.provider) {
        node.data.provider = endpoint.provider;
      }
      if (!node.data.model) {
        node.data.model = modelPreset.name;
      }
    }
  });
}

function hydrateState(snapshot) {
  state.workflowName = snapshot.workflowName || INITIAL_WORKFLOW_NAME;
  state.nodes = Array.isArray(snapshot.nodes) ? snapshot.nodes : [];
  state.edges = Array.isArray(snapshot.edges) ? snapshot.edges : [];
  state.endpointConfigs = Array.isArray(snapshot.endpointConfigs) && snapshot.endpointConfigs.length
    ? snapshot.endpointConfigs.map((endpoint, index) => normalizeEndpointConfig(endpoint, index))
    : getDefaultEndpointConfigs();
  state.selectedNodeId = state.nodes[0]?.id || null;
  state.linkDraft = null;
  state.dragState = null;
  state.sidebarOpen = false;
  state.configOpen = false;
  ensureStepConfigReferences();
}

function loadInitialState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    hydrateState(getSampleState());
    return;
  }

  try {
    hydrateState(JSON.parse(raw));
  } catch (error) {
    console.warn("Failed to parse saved workflow, loading sample.", error);
    hydrateState(getSampleState());
  }
}

function getNode(nodeId) {
  return state.nodes.find((node) => node.id === nodeId) || null;
}

function getIncoming(nodeId) {
  return state.edges.filter((edge) => edge.to === nodeId).map((edge) => edge.from);
}

function getOutgoing(nodeId) {
  return state.edges.filter((edge) => edge.from === nodeId).map((edge) => edge.to);
}

function hasPath(fromId, targetId, visited = new Set()) {
  if (fromId === targetId) {
    return true;
  }
  if (visited.has(fromId)) {
    return false;
  }
  visited.add(fromId);
  return getOutgoing(fromId).some((nextId) => hasPath(nextId, targetId, visited));
}

function canConnect(fromId, toId) {
  if (!fromId || !toId || fromId === toId) {
    return false;
  }

  const fromNode = getNode(fromId);
  const toNode = getNode(toId);
  if (!fromNode || !toNode) {
    return false;
  }

  if (fromNode.type === "output" || toNode.type === "input") {
    return false;
  }

  if (fromNode.type === "input" && toNode.type === "output") {
    return false;
  }

  if (toNode.type === "output" && fromNode.type !== "step") {
    return false;
  }

  const alreadyExists = state.edges.some((edge) => edge.from === fromId && edge.to === toId);
  if (alreadyExists) {
    return false;
  }

  return !hasPath(toId, fromId);
}

function connectNodes(fromId, toId) {
  if (!canConnect(fromId, toId)) {
    return;
  }

  state.edges.push({ id: uid("edge"), from: fromId, to: toId });
  afterStateChange();
}

function getBoardPoint(clientX, clientY) {
  const rect = elements.boardSurface.getBoundingClientRect();
  return {
    x: clientX - rect.left,
    y: clientY - rect.top
  };
}

function applyModelPresetToStep(node, endpointId, modelPresetId) {
  const endpoint = getEndpointConfig(endpointId);
  const modelPreset = getModelPreset(endpointId, modelPresetId);
  if (!node || node.type !== "step" || !endpoint || !modelPreset) {
    return;
  }

  node.data.endpointId = endpoint.id;
  node.data.modelPresetId = modelPreset.id;
  node.data.provider = endpoint.provider;
  node.data.model = modelPreset.name;
  node.data.reasoningMode = modelPreset.reasoningMode;
  node.data.temperature = Number(modelPreset.temperature);
  node.data.maxTokens = Number(modelPreset.maxTokens);
}

function addNode(type, position) {
  const blueprint = nodeBlueprints[type];
  if (!blueprint) {
    return;
  }

  const x = position?.x ?? 180 + state.nodes.length * 26;
  const y = position?.y ?? 180 + state.nodes.length * 24;
  const node = {
    id: uid(type),
    type,
    x: clamp(x, 24, BOARD_SIZE.width - 440),
    y: clamp(y, 24, BOARD_SIZE.height - 340),
    data: blueprint()
  };

  if (type === "step" && state.endpointConfigs[0]) {
    const endpoint = state.endpointConfigs[0];
    const preset = getModelPreset(endpoint.id, endpoint.defaultModelPresetId) || endpoint.models[0];
    if (preset) {
      applyModelPresetToStep(node, endpoint.id, preset.id);
    }
  }

  state.nodes.push(node);
  state.selectedNodeId = node.id;
  afterStateChange();
}

function makeDuplicateValue(value, fallback) {
  const base = String(value || fallback || "").trim();
  if (!base) {
    return `${fallback || "copy"}_${Math.random().toString(36).slice(2, 6)}`;
  }
  return /(_copy| copy)$/i.test(base) ? `${base}_2` : `${base}_copy`;
}

function duplicateNode(nodeId) {
  const source = getNode(nodeId);
  if (!source) {
    return;
  }

  const duplicatedData = structuredClone(source.data);
  duplicatedData.title = `${duplicatedData.title || "Node"} Copy`;

  if (source.type === "input") {
    duplicatedData.key = makeDuplicateValue(duplicatedData.key, "input");
  }

  if (source.type === "step") {
    duplicatedData.outputKey = makeDuplicateValue(duplicatedData.outputKey, "step");
  }

  if (source.type === "output") {
    duplicatedData.key = makeDuplicateValue(duplicatedData.key, "result");
  }

  const duplicatedNode = {
    id: uid(source.type),
    type: source.type,
    x: clamp(source.x + 36, 24, BOARD_SIZE.width - 440),
    y: clamp(source.y + 36, 24, BOARD_SIZE.height - 340),
    data: duplicatedData
  };

  state.nodes.push(duplicatedNode);
  state.selectedNodeId = duplicatedNode.id;
  afterStateChange();
}

function deleteNode(nodeId) {
  state.nodes = state.nodes.filter((node) => node.id !== nodeId);
  state.edges = state.edges.filter((edge) => edge.from !== nodeId && edge.to !== nodeId);
  state.selectedNodeId = state.nodes[0]?.id || null;
  afterStateChange();
}

function beginNodeDrag(event, nodeId) {
  const node = getNode(nodeId);
  if (!node) {
    return;
  }

  state.selectedNodeId = nodeId;
  renderNodeSelection();
  const startPoint = getBoardPoint(event.clientX, event.clientY);
  state.dragState = {
    nodeId,
    offsetX: startPoint.x - node.x,
    offsetY: startPoint.y - node.y
  };

  window.addEventListener("pointermove", handlePointerMove);
  window.addEventListener("pointerup", handlePointerUp, { once: true });
}

function beginLink(event, nodeId) {
  event.stopPropagation();
  const point = getBoardPoint(event.clientX, event.clientY);
  state.linkDraft = {
    from: nodeId,
    x: point.x,
    y: point.y
  };
  window.addEventListener("pointermove", handlePointerMove);
  window.addEventListener("pointerup", handlePointerUp, { once: true });
  renderConnections();
}

function handlePointerMove(event) {
  if (state.dragState) {
    const point = getBoardPoint(event.clientX, event.clientY);
    const node = getNode(state.dragState.nodeId);
    if (!node) {
      return;
    }

    node.x = clamp(point.x - state.dragState.offsetX, 16, BOARD_SIZE.width - 440);
    node.y = clamp(point.y - state.dragState.offsetY, 16, BOARD_SIZE.height - 340);
    const nodeElement = elements.nodesLayer.querySelector(`[data-node-id="${node.id}"]`);
    if (nodeElement) {
      nodeElement.style.left = `${node.x}px`;
      nodeElement.style.top = `${node.y}px`;
    }
    renderConnections();
    return;
  }

  if (state.linkDraft) {
    const point = getBoardPoint(event.clientX, event.clientY);
    state.linkDraft.x = point.x;
    state.linkDraft.y = point.y;
    renderConnections();
  }
}

function handlePointerUp() {
  const movedNode = Boolean(state.dragState);
  state.dragState = null;
  state.linkDraft = null;
  window.removeEventListener("pointermove", handlePointerMove);
  renderConnections();
  if (movedNode) {
    persistState();
  }
}

function renderOptions(options, selected) {
  return options
    .map((option) => `<option value="${option}" ${selected === option ? "selected" : ""}>${escapeHtml(option)}</option>`)
    .join("");
}

function renderEndpointOptions(selected) {
  return state.endpointConfigs
    .map((endpoint) => `<option value="${endpoint.id}" ${selected === endpoint.id ? "selected" : ""}>${escapeHtml(endpoint.provider)} · ${escapeHtml(endpoint.label)}</option>`)
    .join("");
}

function renderModelPresetOptions(endpointId, selected) {
  const endpoint = getEndpointConfig(endpointId);
  if (!endpoint) {
    return `<option value="">No models configured</option>`;
  }

  return endpoint.models
    .map((model) => `<option value="${model.id}" ${selected === model.id ? "selected" : ""}>${escapeHtml(model.label)} · ${escapeHtml(model.name)}</option>`)
    .join("");
}

function getNodeMarkup(node) {
  if (node.type === "input") {
    return `
      <div class="node-form">
        <div class="node-field">
          <label>Title</label>
          <input class="node-title-input" name="title" value="${escapeHtml(node.data.title || "")}">
        </div>
        <div class="node-field">
          <label>Variable key</label>
          <input name="key" value="${escapeHtml(node.data.key || "")}">
        </div>
        <div class="node-field">
          <label>Default value</label>
          <textarea name="defaultValue">${escapeHtml(node.data.defaultValue || "")}</textarea>
        </div>
      </div>
    `;
  }

  if (node.type === "output") {
    return `
      <div class="node-form">
        <div class="node-field">
          <label>Title</label>
          <input class="node-title-input" name="title" value="${escapeHtml(node.data.title || "")}">
        </div>
        <div class="node-field">
          <label>Output key</label>
          <input name="key" value="${escapeHtml(node.data.key || "")}">
        </div>
        <div class="node-row">
          <div class="node-field">
            <label>Format</label>
            <select name="format">
              ${renderOptions(["json", "markdown", "text"], node.data.format)}
            </select>
          </div>
          <div class="node-field">
            <label>Schema hint</label>
            <input name="schemaHint" value="${escapeHtml(node.data.schemaHint || "")}">
          </div>
        </div>
      </div>
    `;
  }

  return `
    <div class="node-form">
      <div class="node-field">
        <label>Title</label>
        <input class="node-title-input" name="title" value="${escapeHtml(node.data.title || "")}">
      </div>
      <div class="node-row">
        <div class="node-field">
          <label>Provider</label>
          <select name="endpointId">
            ${renderEndpointOptions(node.data.endpointId)}
          </select>
        </div>
        <div class="node-field">
          <label>Model</label>
          <select name="modelPresetId">
            ${renderModelPresetOptions(node.data.endpointId, node.data.modelPresetId)}
          </select>
        </div>
      </div>
      <div class="node-field">
        <label>Prompt</label>
        <textarea name="prompt">${escapeHtml(node.data.prompt || "")}</textarea>
      </div>
      <div class="node-row">
        <div class="node-field">
          <label>Reason mode</label>
          <select name="reasoningMode">
            ${renderOptions(["off", "on"], node.data.reasoningMode)}
          </select>
        </div>
        <div class="node-field">
          <label>Output key</label>
          <input name="outputKey" value="${escapeHtml(node.data.outputKey || "")}">
        </div>
      </div>
      <div class="node-row">
        <div class="node-field">
          <label>Temperature</label>
          <input name="temperature" type="number" step="0.1" min="0" max="2" value="${escapeHtml(String(node.data.temperature ?? 0.4))}">
        </div>
        <div class="node-field">
          <label>Max tokens</label>
          <input name="maxTokens" type="number" min="1" max="32000" value="${escapeHtml(String(node.data.maxTokens ?? 700))}">
        </div>
      </div>
      <label class="node-toggle">
        <input name="recursionEnabled" type="checkbox" ${node.data.recursionEnabled ? "checked" : ""}>
        <span>Recursive refinement</span>
      </label>
      <div class="recursive-fields">
        <div class="node-row">
          <div class="node-field">
            <label>Max iterations</label>
            <input name="maxIterations" type="number" min="1" max="12" value="${escapeHtml(String(node.data.maxIterations ?? 3))}">
          </div>
          <div class="node-field">
            <label>Stop condition</label>
            <input name="stopCondition" value="${escapeHtml(node.data.stopCondition || "")}">
          </div>
        </div>
        <div class="node-field">
          <label>Recursive critique prompt</label>
          <textarea name="recursionPrompt">${escapeHtml(node.data.recursionPrompt || "")}</textarea>
        </div>
      </div>
    </div>
  `;
}

function renderNodes() {
  elements.nodesLayer.innerHTML = "";
  const fragment = document.createDocumentFragment();

  state.nodes.forEach((node) => {
    const article = document.createElement("article");
    article.className = "workflow-node";
    article.dataset.nodeId = node.id;
    article.dataset.type = node.type;
    article.dataset.recursive = node.type === "step" && node.data.recursionEnabled ? "true" : "false";
    article.style.left = `${node.x}px`;
    article.style.top = `${node.y}px`;
    article.innerHTML = `
      <button class="node-handle input-handle" type="button" aria-label="Connect input"></button>
      <div class="node-shell">
        <div class="node-toolbar">
          <button class="node-grip" type="button">Drag</button>
          <div class="node-meta">
            <span class="node-type-badge">${escapeHtml(node.type)}</span>
            <span class="node-id">${escapeHtml(node.id.replace(/^[^_]+_/, ""))}</span>
          </div>
          <button class="node-copy" type="button">Copy</button>
          <button class="node-delete" type="button">Delete</button>
        </div>
        ${getNodeMarkup(node)}
      </div>
      <button class="node-handle output-handle" type="button" aria-label="Connect output"></button>
    `;

    const inputHandle = article.querySelector(".input-handle");
    const outputHandle = article.querySelector(".output-handle");

    if (node.type === "input") {
      inputHandle.style.visibility = "hidden";
      inputHandle.disabled = true;
    }

    if (node.type === "output") {
      outputHandle.style.visibility = "hidden";
      outputHandle.disabled = true;
    }

    article.querySelector(".node-grip").addEventListener("pointerdown", (event) => beginNodeDrag(event, node.id));
    article.querySelector(".node-copy").addEventListener("click", (event) => {
      event.stopPropagation();
      duplicateNode(node.id);
    });
    article.querySelector(".node-delete").addEventListener("click", (event) => {
      event.stopPropagation();
      deleteNode(node.id);
    });

    article.addEventListener("click", () => selectNode(node.id));
    article.addEventListener("focusin", () => selectNode(node.id));

    outputHandle.addEventListener("pointerdown", (event) => beginLink(event, node.id));
    inputHandle.addEventListener("pointerup", (event) => {
      if (!state.linkDraft) {
        return;
      }
      event.stopPropagation();
      connectNodes(state.linkDraft.from, node.id);
      state.linkDraft = null;
      window.removeEventListener("pointermove", handlePointerMove);
      renderConnections();
    });

    article.querySelectorAll("[name]").forEach((field) => {
      field.addEventListener("input", (event) => updateNodeField(node.id, event.target.name, event.target));
      field.addEventListener("change", (event) => updateNodeField(node.id, event.target.name, event.target, true));
    });

    fragment.appendChild(article);
  });

  elements.nodesLayer.appendChild(fragment);
  renderNodeSelection();
}

function selectNode(nodeId) {
  if (state.selectedNodeId === nodeId) {
    return;
  }
  state.selectedNodeId = nodeId;
  renderNodeSelection();
}

function renderNodeSelection() {
  elements.nodesLayer.querySelectorAll(".workflow-node").forEach((nodeElement) => {
    nodeElement.classList.toggle("is-selected", nodeElement.dataset.nodeId === state.selectedNodeId);
  });
}

function updateNodeField(nodeId, fieldName, field) {
  const node = getNode(nodeId);
  if (!node) {
    return;
  }

  if (field.type === "checkbox") {
    node.data[fieldName] = field.checked;
  } else if (field.type === "number") {
    node.data[fieldName] = field.value === "" ? 0 : Number(field.value);
  } else {
    node.data[fieldName] = field.value;
  }

  if (fieldName === "endpointId") {
    const endpoint = getEndpointConfig(node.data.endpointId);
    const modelPreset = endpoint
      ? getModelPreset(endpoint.id, endpoint.defaultModelPresetId) || endpoint.models[0]
      : null;
    if (endpoint && modelPreset) {
      applyModelPresetToStep(node, endpoint.id, modelPreset.id);
    }
    renderNodes();
    renderConnections();
  } else if (fieldName === "modelPresetId") {
    const endpoint = getEndpointConfig(node.data.endpointId);
    const modelPreset = endpoint
      ? getModelPreset(endpoint.id, node.data.modelPresetId) || endpoint.models[0]
      : null;
    if (endpoint && modelPreset) {
      applyModelPresetToStep(node, endpoint.id, modelPreset.id);
    }
    renderNodes();
    renderConnections();
  } else if (fieldName === "recursionEnabled") {
    const nodeElement = field.closest(".workflow-node");
    if (nodeElement) {
      nodeElement.dataset.recursive = field.checked ? "true" : "false";
    }
  }

  renderStatus();
  persistState();
}

function renderEndpointConfigs() {
  elements.endpointConfigList.innerHTML = "";

  if (!state.endpointConfigs.length) {
    elements.endpointConfigList.innerHTML = '<div class="empty-state">No endpoints yet. Add one and define model presets.</div>';
    return;
  }

  const fragment = document.createDocumentFragment();

  state.endpointConfigs.forEach((endpoint) => {
    const article = document.createElement("article");
    article.className = "config-card endpoint-card";
    article.dataset.endpointId = endpoint.id;
    article.innerHTML = `
      <div class="endpoint-header">
        <div>
          <h4 class="endpoint-title">${escapeHtml(endpoint.label)}</h4>
          <div class="stack-note">Provider profile shared by all workflows in this browser.</div>
        </div>
        <button class="endpoint-delete" type="button">Delete provider</button>
      </div>
      <div class="form-row">
        <label class="config-card">
          <span class="form-label">Label</span>
          <input data-endpoint-field="label" value="${escapeHtml(endpoint.label)}">
        </label>
        <label class="config-card">
          <span class="form-label">Provider</span>
          <select data-endpoint-field="provider">
            ${renderOptions(["openai", "anthropic", "google", "custom"], endpoint.provider)}
          </select>
        </label>
      </div>
      <label class="config-card">
        <span class="form-label">API key</span>
        <input data-endpoint-field="apiKey" type="password" value="${escapeHtml(endpoint.apiKey)}" placeholder="Stored locally only">
      </label>
      <label class="config-card">
        <span class="form-label">Default model</span>
        <select data-endpoint-field="defaultModelPresetId">
          ${endpoint.models.map((model) => `<option value="${model.id}" ${endpoint.defaultModelPresetId === model.id ? "selected" : ""}>${escapeHtml(model.label)}</option>`).join("")}
        </select>
      </label>
      <div class="inline-actions">
        <button class="subtle-button" type="button" data-add-model-preset="${endpoint.id}">Add model</button>
      </div>
      <div class="model-preset-list">
        ${endpoint.models.map((model) => `
          <article class="config-card model-preset-card" data-model-id="${model.id}">
            <div class="card-toolbar">
              <div>
                <h5 class="model-title">${escapeHtml(model.label)}</h5>
                <div class="stack-note">Reusable model defaults for this provider.</div>
              </div>
              <button class="model-delete" type="button" data-delete-model="${model.id}">Delete model</button>
            </div>
            <div class="form-row">
              <label class="config-card">
                <span class="form-label">Preset label</span>
                <input data-model-field="label" data-model-id="${model.id}" value="${escapeHtml(model.label)}">
              </label>
              <label class="config-card">
                <span class="form-label">Model name</span>
                <input data-model-field="name" data-model-id="${model.id}" value="${escapeHtml(model.name)}">
              </label>
            </div>
            <div class="form-row">
              <label class="config-card">
                <span class="form-label">Reason default</span>
                <select data-model-field="reasoningMode" data-model-id="${model.id}">
                  ${renderOptions(["off", "on"], model.reasoningMode)}
                </select>
              </label>
              <label class="config-card">
                <span class="form-label">Temperature</span>
                <input data-model-field="temperature" data-model-id="${model.id}" type="number" step="0.1" min="0" max="2" value="${escapeHtml(String(model.temperature))}">
              </label>
            </div>
            <label class="config-card">
              <span class="form-label">Max tokens</span>
              <input data-model-field="maxTokens" data-model-id="${model.id}" type="number" min="1" max="32000" value="${escapeHtml(String(model.maxTokens))}">
            </label>
          </article>
        `).join("")}
      </div>
    `;

    article.querySelector(".endpoint-delete").addEventListener("click", () => deleteEndpointConfig(endpoint.id));
    article.querySelectorAll("[data-endpoint-field]").forEach((field) => {
      field.addEventListener("input", (event) => updateEndpointField(endpoint.id, event.target.dataset.endpointField, event.target));
      field.addEventListener("change", (event) => updateEndpointField(endpoint.id, event.target.dataset.endpointField, event.target, true));
    });
    article.querySelector(`[data-add-model-preset="${endpoint.id}"]`).addEventListener("click", () => addModelPreset(endpoint.id));
    article.querySelectorAll("[data-model-field]").forEach((field) => {
      field.addEventListener("input", (event) => updateModelPresetField(endpoint.id, event.target.dataset.modelId, event.target.dataset.modelField, event.target));
      field.addEventListener("change", (event) => updateModelPresetField(endpoint.id, event.target.dataset.modelId, event.target.dataset.modelField, event.target, true));
    });
    article.querySelectorAll("[data-delete-model]").forEach((button) => {
      button.addEventListener("click", () => deleteModelPreset(endpoint.id, button.dataset.deleteModel));
    });

    fragment.appendChild(article);
  });

  elements.endpointConfigList.appendChild(fragment);
}

function updateEndpointField(endpointId, fieldName, field, shouldRefresh = false) {
  const endpoint = getEndpointConfig(endpointId);
  if (!endpoint) {
    return;
  }

  endpoint[fieldName] = field.value;

  if (fieldName === "provider") {
    state.nodes.forEach((node) => {
      if (node.type === "step" && node.data.endpointId === endpointId) {
        node.data.provider = endpoint.provider;
      }
    });
  }

  persistState();

  if (shouldRefresh) {
    ensureStepConfigReferences();
    render();
  }
}

function addEndpointConfig() {
  const endpoint = createEndpointConfig({
    label: `Provider ${state.endpointConfigs.length + 1}`,
    models: [
      createModelPreset({
        label: "Default preset"
      })
    ]
  });
  endpoint.defaultModelPresetId = endpoint.models[0].id;
  state.endpointConfigs.push(endpoint);
  ensureStepConfigReferences();
  persistState();
  render();
}

function deleteEndpointConfig(endpointId) {
  state.endpointConfigs = state.endpointConfigs.filter((endpoint) => endpoint.id !== endpointId);
  if (!state.endpointConfigs.length) {
    state.endpointConfigs = getDefaultEndpointConfigs();
  }
  ensureStepConfigReferences();
  persistState();
  render();
}

function addModelPreset(endpointId) {
  const endpoint = getEndpointConfig(endpointId);
  if (!endpoint) {
    return;
  }

  const preset = createModelPreset({
    label: `Preset ${endpoint.models.length + 1}`
  });
  endpoint.models.push(preset);
  if (!endpoint.defaultModelPresetId) {
    endpoint.defaultModelPresetId = preset.id;
  }
  ensureStepConfigReferences();
  persistState();
  render();
}

function deleteModelPreset(endpointId, modelPresetId) {
  const endpoint = getEndpointConfig(endpointId);
  if (!endpoint) {
    return;
  }

  endpoint.models = endpoint.models.filter((model) => model.id !== modelPresetId);
  if (!endpoint.models.length) {
    endpoint.models = [createModelPreset({ label: "Default preset" })];
  }
  if (!endpoint.models.some((model) => model.id === endpoint.defaultModelPresetId)) {
    endpoint.defaultModelPresetId = endpoint.models[0].id;
  }
  ensureStepConfigReferences();
  persistState();
  render();
}

function updateModelPresetField(endpointId, modelPresetId, fieldName, field, shouldRefresh = false) {
  const modelPreset = getModelPreset(endpointId, modelPresetId);
  if (!modelPreset) {
    return;
  }

  if (field.type === "number") {
    modelPreset[fieldName] = field.value === "" ? 0 : Number(field.value);
  } else {
    modelPreset[fieldName] = field.value;
  }

  persistState();

  if (shouldRefresh) {
    ensureStepConfigReferences();
    render();
  }
}

function getHandleCenter(nodeId, selector) {
  const nodeElement = elements.nodesLayer.querySelector(`[data-node-id="${nodeId}"]`);
  if (!nodeElement) {
    return null;
  }

  const handle = nodeElement.querySelector(selector);
  if (!handle || handle.disabled) {
    return null;
  }

  const handleRect = handle.getBoundingClientRect();
  const surfaceRect = elements.boardSurface.getBoundingClientRect();

  return {
    x: handleRect.left - surfaceRect.left + handleRect.width / 2,
    y: handleRect.top - surfaceRect.top + handleRect.height / 2
  };
}

function makeCurvePath(start, end) {
  const distance = Math.max(80, Math.abs(end.x - start.x) * 0.55);
  return `M ${start.x} ${start.y} C ${start.x + distance} ${start.y}, ${end.x - distance} ${end.y}, ${end.x} ${end.y}`;
}

function renderConnections() {
  elements.connectionsLayer.innerHTML = "";

  state.edges.forEach((edge) => {
    const start = getHandleCenter(edge.from, ".output-handle");
    const end = getHandleCenter(edge.to, ".input-handle");
    if (!start || !end) {
      return;
    }

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", "connection-line");
    path.setAttribute("d", makeCurvePath(start, end));
    elements.connectionsLayer.appendChild(path);
  });

  if (state.linkDraft) {
    const start = getHandleCenter(state.linkDraft.from, ".output-handle");
    if (!start) {
      return;
    }
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", "connection-line draft");
    path.setAttribute("d", makeCurvePath(start, { x: state.linkDraft.x, y: state.linkDraft.y }));
    elements.connectionsLayer.appendChild(path);
  }
}

function topologicalSort() {
  const incomingCount = new Map(state.nodes.map((node) => [node.id, 0]));
  const outgoingMap = new Map(state.nodes.map((node) => [node.id, []]));

  state.edges.forEach((edge) => {
    incomingCount.set(edge.to, (incomingCount.get(edge.to) || 0) + 1);
    outgoingMap.get(edge.from)?.push(edge.to);
  });

  const queue = state.nodes
    .filter((node) => (incomingCount.get(node.id) || 0) === 0)
    .map((node) => node.id);
  const ordered = [];

  while (queue.length) {
    const current = queue.shift();
    ordered.push(current);
    (outgoingMap.get(current) || []).forEach((next) => {
      incomingCount.set(next, (incomingCount.get(next) || 0) - 1);
      if ((incomingCount.get(next) || 0) === 0) {
        queue.push(next);
      }
    });
  }

  return {
    ordered,
    hasCycle: ordered.length !== state.nodes.length
  };
}

function collectAncestors(nodeId, visited = new Set()) {
  getIncoming(nodeId).forEach((parentId) => {
    if (visited.has(parentId)) {
      return;
    }
    visited.add(parentId);
    collectAncestors(parentId, visited);
  });
  return visited;
}

function collectDescendants(nodeId, visited = new Set()) {
  getOutgoing(nodeId).forEach((childId) => {
    if (visited.has(childId)) {
      return;
    }
    visited.add(childId);
    collectDescendants(childId, visited);
  });
  return visited;
}

function getAvailableContextKeys(stepNode) {
  const ancestorIds = collectAncestors(stepNode.id);
  const ancestorNodes = Array.from(ancestorIds)
    .map((id) => getNode(id))
    .filter(Boolean);

  return new Set(
    ancestorNodes.flatMap((node) => {
      if (node.type === "input") {
        return [node.data.key];
      }
      if (node.type === "step") {
        return [node.data.outputKey];
      }
      return [];
    })
  );
}

function getEndpointExport(endpointId) {
  const endpoint = getEndpointConfig(endpointId);
  if (!endpoint) {
    return null;
  }

  return {
    id: endpoint.id,
    label: endpoint.label,
    provider: endpoint.provider,
    defaultModelPresetId: endpoint.defaultModelPresetId,
    auth: {
      apiKeyStoredLocally: Boolean(endpoint.apiKey),
      apiKeyRef: endpoint.apiKey ? `local:${endpoint.id}` : null
    },
    models: endpoint.models.map((model) => ({
      id: model.id,
      label: model.label,
      name: model.name,
      reasoningMode: model.reasoningMode,
      temperature: Number(model.temperature),
      maxTokens: Number(model.maxTokens)
    }))
  };
}

function getExecutionConfig(node) {
  if (node.type === "input") {
    return {
      key: node.data.key,
      defaultValue: node.data.defaultValue
    };
  }

  if (node.type === "step") {
    return {
      providerConfigId: node.data.endpointId || null,
      providerConfig: getEndpointExport(node.data.endpointId),
      modelPresetId: node.data.modelPresetId || null,
      prompt: node.data.prompt,
      variables: extractVariables(node.data.prompt),
      model: {
        provider: node.data.provider,
        name: node.data.model,
        reasoningMode: node.data.reasoningMode,
        temperature: Number(node.data.temperature),
        maxTokens: Number(node.data.maxTokens)
      },
      recursion: {
        enabled: Boolean(node.data.recursionEnabled),
        maxIterations: Math.max(1, Number(node.data.maxIterations || 1)),
        stopCondition: node.data.stopCondition,
        recursionPrompt: node.data.recursionPrompt
      },
      outputKey: node.data.outputKey
    };
  }

  return {
    key: node.data.key,
    format: node.data.format,
    schemaHint: node.data.schemaHint
  };
}

function compileWorkflow() {
  const topo = topologicalSort();
  const orderedNodes = topo.ordered.map((id) => getNode(id)).filter(Boolean);
  const warnings = [];

  if (topo.hasCycle) {
    warnings.push("Cycle detected: execution order is incomplete until the loop is removed.");
  }

  state.nodes.forEach((node) => {
    if (node.type === "step") {
      if (!node.data.prompt?.trim()) {
        warnings.push(`${node.data.title} has no prompt text.`);
      }
      if (!node.data.model?.trim()) {
        warnings.push(`${node.data.title} has no model selected.`);
      }
      if (getOutgoing(node.id).length === 0) {
        warnings.push(`${node.data.title} is not wired to another step or output.`);
      }
      if (node.data.recursionEnabled && !node.data.recursionPrompt?.trim()) {
        warnings.push(`${node.data.title} enables recursion but has no recursive critique prompt.`);
      }
      if (!node.data.endpointId) {
        warnings.push(`${node.data.title} has no provider selected.`);
      } else {
        const endpoint = getEndpointConfig(node.data.endpointId);
        const preset = getModelPreset(node.data.endpointId, node.data.modelPresetId);
        if (!endpoint) {
          warnings.push(`${node.data.title} points to a missing provider profile.`);
        } else {
          if (!endpoint.apiKey?.trim()) {
            warnings.push(`${node.data.title} uses a provider profile without an API key.`);
          }
          if (!preset) {
            warnings.push(`${node.data.title} points to a missing model preset.`);
          }
        }
      }

      const availableContextKeys = getAvailableContextKeys(node);
      const missingVariables = extractVariables(node.data.prompt).filter((key) => !availableContextKeys.has(key));
      if (missingVariables.length) {
        warnings.push(`${node.data.title} references missing variables: ${missingVariables.join(", ")}.`);
      }
    }

    if (node.type === "output" && getIncoming(node.id).length === 0) {
      warnings.push(`${node.data.title} is not connected to any step result.`);
    }
  });

  const inputs = state.nodes
    .filter((node) => node.type === "input")
    .map((node) => ({
      id: node.id,
      key: node.data.key,
      title: node.data.title,
      defaultValue: node.data.defaultValue
    }));

  const steps = orderedNodes.map((node) => ({
    id: node.id,
    type: node.type,
    title: node.data.title,
    dependsOn: getIncoming(node.id),
    config: getExecutionConfig(node)
  }));

  const reusableCalls = orderedNodes
    .filter((node) => node.type === "step")
    .map((node) => {
      const directInputs = getIncoming(node.id)
        .map((id) => getNode(id))
        .filter(Boolean)
        .map((upstreamNode) => {
          if (upstreamNode.type === "input") {
            return {
              sourceNode: upstreamNode.id,
              sourceType: "input",
              key: upstreamNode.data.key
            };
          }

          return {
            sourceNode: upstreamNode.id,
            sourceType: "step",
            key: upstreamNode.data.outputKey
          };
        });

      const downstreamTargets = Array.from(collectDescendants(node.id))
        .map((id) => getNode(id))
        .filter((item) => item && item.type === "output")
        .map((item) => ({
          key: item.data.key,
          format: item.data.format,
          schemaHint: item.data.schemaHint,
          nodeId: item.id
        }));

      return {
        id: `call_${node.id}`,
        title: node.data.title,
        sourceNode: node.id,
        providerConfigId: node.data.endpointId || null,
        providerConfig: getEndpointExport(node.data.endpointId),
        modelPresetId: node.data.modelPresetId || null,
        inputBindings: directInputs,
        prompt: {
          template: node.data.prompt,
          variables: extractVariables(node.data.prompt)
        },
        model: {
          provider: node.data.provider,
          name: node.data.model,
          reasoningMode: node.data.reasoningMode,
          temperature: Number(node.data.temperature),
          maxTokens: Number(node.data.maxTokens)
        },
        recursion: {
          enabled: Boolean(node.data.recursionEnabled),
          maxIterations: Math.max(1, Number(node.data.maxIterations || 1)),
          stopCondition: node.data.stopCondition,
          recursionPrompt: node.data.recursionPrompt
        },
        outputKey: node.data.outputKey,
        downstreamTargets
      };
    });

  return {
    name: state.workflowName,
    version: "5.0",
    exportedAt: new Date().toISOString(),
    purpose: "Prompt engineering workflow with recursive LLM steps",
    settings: {
      llmProviders: state.endpointConfigs.map((endpoint) => getEndpointExport(endpoint.id))
    },
    inputs,
    workflow: {
      nodes: state.nodes.map((node) => ({
        id: node.id,
        type: node.type,
        position: { x: Math.round(node.x), y: Math.round(node.y) },
        data: node.data
      })),
      edges: state.edges.map((edge) => ({
        id: edge.id,
        from: edge.from,
        to: edge.to
      }))
    },
    execution: {
      status: topo.hasCycle ? "needs-fix" : "ready",
      steps,
      reusableCalls
    },
    validation: {
      warnings
    }
  };
}

function renderStatus() {
  const compiled = compileWorkflow();
  const warningCount = compiled.validation.warnings.length;
  elements.workflowTitle.textContent = state.workflowName;
  if (elements.workflowNameInput.value !== state.workflowName) {
    elements.workflowNameInput.value = state.workflowName;
  }
  const modelCount = state.endpointConfigs.reduce((sum, endpoint) => sum + endpoint.models.length, 0);
  elements.graphMeta.textContent = `${state.nodes.length} nodes · ${state.edges.length} connections · ${state.endpointConfigs.length} providers · ${modelCount} models`;
  elements.graphStatus.textContent = warningCount ? `${warningCount} warning${warningCount > 1 ? "s" : ""}` : "Ready";
  elements.graphStatus.classList.toggle("is-good", warningCount === 0);
  elements.graphStatus.title = compiled.validation.warnings.join("\n");
}

function renderSidebar() {
  elements.leftSidebar.classList.toggle("is-collapsed", !state.sidebarOpen);
  elements.toggleSidebarButton.textContent = state.sidebarOpen ? "Hide tools" : "Tools";
}

function renderConfigOverlay() {
  elements.configOverlay.classList.toggle("is-hidden", !state.configOpen);
  elements.configOverlay.setAttribute("aria-hidden", String(!state.configOpen));
}

function render() {
  renderSidebar();
  renderConfigOverlay();
  renderEndpointConfigs();
  renderNodes();
  renderConnections();
  renderStatus();
}

function afterStateChange() {
  render();
  persistState();
}

function copyText(text) {
  if (navigator.clipboard?.writeText) {
    return navigator.clipboard.writeText(text);
  }
  return Promise.reject(new Error("Clipboard API unavailable"));
}

async function copyWorkflowJson(compact = false) {
  const payload = compact
    ? JSON.stringify(compileWorkflow())
    : JSON.stringify(compileWorkflow(), null, 2);

  try {
    await copyText(payload);
  } catch (error) {
    alert("Clipboard access failed. Try the download button instead.");
  }
}

function slugify(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "prompt-workflow";
}

function downloadJson() {
  const compiled = compileWorkflow();
  const blob = new Blob([JSON.stringify(compiled, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${slugify(state.workflowName || "prompt-workflow")}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function importWorkflow(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const parsed = JSON.parse(reader.result);
      const importedEndpoints = parsed.endpointConfigs
        || parsed.settings?.llmProviders
        || parsed.settings?.llmEndpoints
        || [];
      const snapshot = parsed.workflow
        ? {
            workflowName: parsed.name || INITIAL_WORKFLOW_NAME,
            nodes: parsed.workflow.nodes.map((node) => ({
              id: node.id,
              type: node.type,
              x: node.position?.x ?? 100,
              y: node.position?.y ?? 100,
              data: node.data || {}
            })),
            edges: parsed.workflow.edges || [],
            endpointConfigs: importedEndpoints.map((endpoint, index) => normalizeEndpointConfig(endpoint, index))
          }
        : parsed;
      hydrateState(snapshot);
      afterStateChange();
    } catch (error) {
      alert("That file is not a valid workflow JSON.");
    }
  };
  reader.readAsText(file);
}

function wireUi() {
  document.querySelectorAll("[data-node-type]").forEach((button) => {
    button.addEventListener("click", () => addNode(button.dataset.nodeType));
  });

  elements.workflowNameInput.addEventListener("input", (event) => {
    state.workflowName = event.target.value.trim() || INITIAL_WORKFLOW_NAME;
    renderStatus();
    persistState();
  });

  elements.toggleSidebarButton.addEventListener("click", () => {
    state.sidebarOpen = !state.sidebarOpen;
    renderSidebar();
  });

  elements.closeSidebarButton.addEventListener("click", () => {
    state.sidebarOpen = false;
    renderSidebar();
  });

  elements.openConfigButton.addEventListener("click", () => {
    state.configOpen = true;
    renderConfigOverlay();
  });

  elements.closeConfigButton.addEventListener("click", () => {
    state.configOpen = false;
    renderConfigOverlay();
  });

  elements.configOverlay.addEventListener("click", (event) => {
    if (event.target === elements.configOverlay) {
      state.configOpen = false;
      renderConfigOverlay();
    }
  });

  elements.addEndpointButton.addEventListener("click", addEndpointConfig);

  document.getElementById("copyJsonButton").addEventListener("click", () => copyWorkflowJson(false));
  document.getElementById("copyCompactButton").addEventListener("click", () => copyWorkflowJson(true));
  document.getElementById("downloadJsonButton").addEventListener("click", downloadJson);
  document.getElementById("importJsonButton").addEventListener("click", () => elements.importJsonInput.click());
  document.getElementById("resetBoardButton").addEventListener("click", () => {
    hydrateState(getSampleState());
    afterStateChange();
  });
  document.getElementById("loadSampleButton").addEventListener("click", () => {
    hydrateState(getSampleState());
    afterStateChange();
  });

  elements.importJsonInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) {
      importWorkflow(file);
    }
    event.target.value = "";
  });

  elements.boardViewport.addEventListener("scroll", renderConnections);
  elements.boardSurface.addEventListener("click", (event) => {
    if (event.target.closest(".workflow-node")) {
      return;
    }
    state.selectedNodeId = null;
    renderNodeSelection();
  });
  elements.boardSurface.addEventListener("dblclick", (event) => {
    if (event.target.closest(".workflow-node")) {
      return;
    }
    addNode("step", getBoardPoint(event.clientX, event.clientY));
  });

  window.addEventListener("resize", renderConnections);
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.configOpen) {
      state.configOpen = false;
      renderConfigOverlay();
      return;
    }

    if ((event.key === "Delete" || event.key === "Backspace") && state.selectedNodeId) {
      const targetTag = document.activeElement?.tagName;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(targetTag)) {
        return;
      }
      deleteNode(state.selectedNodeId);
    }
  });
}

loadInitialState();
wireUi();
render();
