/**
 * Factor Graph Module
 * 
 * A visual drag-and-drop interface for creating hierarchical state factors.
 * This module handles the visualization of factor relationships and
 * automatically generates the correct hierarchical structure.
 */

class FactorGraph {
  constructor(containerId, onFactorSelect) {
    this.containerId = containerId;
    this.onFactorSelect = onFactorSelect || function() {};
    this.nodes = [];
    this.edges = [];
    this.selectedNode = null;
    this.draggedNode = null;
    this.dragOffsetX = 0;
    this.dragOffsetY = 0;
    this.canvas = null;
    this.ctx = null;
    this.factorTypes = {
      'continuous': '#4a6cf7',
      'categorical': '#10b981',
      'discrete': '#ef4444'
    };
    
    this.init();
  }
  
  /**
   * Initialize the graph
   */
  init() {
    try {
      console.log('Initializing factor graph');
      
      // Make sure the container exists
      this.container = document.getElementById(this.containerId);
      if (!this.container) {
        console.error('Container not found:', this.containerId);
        return;
      }
      
      // Create canvas element
      this.canvas = document.createElement('canvas');
      this.canvas.style.width = '100%';
      this.canvas.style.height = '100%';
      this.container.appendChild(this.canvas);
      
      // Force a specific size to ensure it's visible
      this.canvas.width = this.container.offsetWidth || 800;
      this.canvas.height = this.container.offsetHeight || 400;
      
      // Get the context
      this.ctx = this.canvas.getContext('2d');
      if (!this.ctx) {
        console.error('Could not get 2D context');
        return;
      }
      
      // Set up event listeners
      this.addEventListeners();
      
      console.log(`Canvas initialized with size: ${this.canvas.width}x${this.canvas.height}`);
      this.render();
    } catch (e) {
      console.error('Error initializing factor graph:', e);
    }
  }
  
  /**
   * Add event listeners to the canvas
   */
  addEventListeners() {
    // Add event listeners for mouse interactions
    this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
    this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
    this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
    this.canvas.addEventListener('dblclick', this.handleDoubleClick.bind(this));
    
    // Add button to container
    const addButton = document.createElement('button');
    addButton.className = 'factor-graph-add-btn';
    addButton.textContent = 'Add Factor';
    addButton.style.position = 'absolute';
    addButton.style.top = '10px';
    addButton.style.right = '10px';
    addButton.style.zIndex = '100';
    addButton.style.padding = '8px 16px';
    addButton.style.backgroundColor = '#3b82f6';
    addButton.style.color = 'white';
    addButton.style.border = 'none';
    addButton.style.borderRadius = '4px';
    addButton.style.cursor = 'pointer';
    addButton.style.fontWeight = '500';
    
    addButton.addEventListener('click', () => {
      this.quickAddFactor();
    });
    
    this.container.appendChild(addButton);
    
    // Add layout button
    const layoutButton = document.createElement('button');
    layoutButton.className = 'factor-graph-layout-btn';
    layoutButton.textContent = 'Auto Layout';
    layoutButton.style.position = 'absolute';
    layoutButton.style.top = '10px';
    layoutButton.style.right = '120px';
    layoutButton.style.zIndex = '100';
    layoutButton.style.padding = '8px 16px';
    layoutButton.style.backgroundColor = '#64748b';
    layoutButton.style.color = 'white';
    layoutButton.style.border = 'none';
    layoutButton.style.borderRadius = '4px';
    layoutButton.style.cursor = 'pointer';
    layoutButton.style.fontWeight = '500';
    
    layoutButton.addEventListener('click', () => {
      this.autoLayout();
      this.render();
    });
    
    this.container.appendChild(layoutButton);
    
    // Add window resize listener
    window.addEventListener('resize', () => {
      this.resize();
    });
  }
  
  /**
   * Resize the canvas to match the container
   */
  resize() {
    if (!this.canvas || !this.container) {
      console.warn('Cannot resize: canvas or container not initialized');
      return;
    }
    
    try {
      console.log('Resizing factor graph');
      
      // Get the container dimensions
      const rect = this.container.getBoundingClientRect();
      
      // Ensure canvas dimensions match the container
      this.canvas.width = rect.width;
      this.canvas.height = rect.height || 400; // Minimum height
      
      console.log(`Resized canvas to ${this.canvas.width}x${this.canvas.height}`);
      
      // Re-render the graph
      this.render();
    } catch (e) {
      console.error('Error resizing factor graph:', e);
    }
  }
  
  /**
   * Handle mouse down event
   */
  handleMouseDown(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    // Check if a node was clicked
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const node = this.nodes[i];
      const dx = mouseX - node.x;
      const dy = mouseY - node.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance <= node.radius) {
        this.selectedNode = node;
        this.draggedNode = node;
        this.dragOffsetX = dx;
        this.dragOffsetY = dy;
        this.render();
        return;
      }
    }
    
    // Deselect if clicking on empty space
    this.selectedNode = null;
    this.render();
  }
  
  /**
   * Handle mouse move event
   */
  handleMouseMove(e) {
    if (!this.draggedNode) return;
    
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    this.draggedNode.x = mouseX - this.dragOffsetX;
    this.draggedNode.y = mouseY - this.dragOffsetY;
    
    this.render();
  }
  
  /**
   * Handle mouse up event
   */
  handleMouseUp() {
    this.draggedNode = null;
  }
  
  /**
   * Handle double click event - directly edit a node when double-clicked
   */
  handleDoubleClick(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    // Check if a node was double-clicked
    for (const node of this.nodes) {
      const dx = mouseX - node.x;
      const dy = mouseY - node.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance <= node.radius) {
        // Edit the node - trigger callback to open editor
        if (this.onFactorSelect) {
          this.onFactorSelect(node);
        }
        return;
      }
      
      // Check if the "add child" button was clicked
      if (node === this.selectedNode) {
        const addButtonX = node.x + node.radius * 0.7;
        const addButtonY = node.y - node.radius * 0.7;
        const addButtonDistance = Math.sqrt(
          Math.pow(mouseX - addButtonX, 2) + 
          Math.pow(mouseY - addButtonY, 2)
        );
        
        if (addButtonDistance <= 10) {
          // Add a child node for this selected node
          // Create new factor as a child
          const factorCount = this.nodes.length;
          const timestamp = new Date().getTime().toString().slice(-4);
          const factorName = `child_${factorCount + 1}_${timestamp}`;
          
          // Create a basic continuous factor as child
          const factor = {
            name: factorName,
            factor_type: 'continuous',
            value: 0,
            params: {
              loc: 0,
              scale: 1
            },
            relationships: {
              depends_on: [node.id]
            }
          };
          
          // Add below the parent node
          const x = node.x + Math.random() * 50 - 25;
          const y = node.y + 100 + Math.random() * 30;
          
          // Add to graph
          this.addFactor(factor, x, y);
          return;
        }
      }
    }
  }
  
  /**
   * Add a factor node to the graph
   */
  addFactor(factor, x, y) {
    const node = {
      id: factor.name,
      label: factor.name,
      type: factor.factor_type,
      x: x || this.canvas.width / 2,
      y: y || this.canvas.height / 3,
      radius: 30,
      factor: factor,
      children: []
    };
    
    this.nodes.push(node);
    
    // Add edges for parent-child relationships
    if (factor.relationships && factor.relationships.depends_on) {
      for (const parentId of factor.relationships.depends_on) {
        const parentNode = this.findNodeById(parentId);
        if (parentNode) {
          this.edges.push({
            from: parentNode,
            to: node
          });
          
          // Add to parent's children list
          parentNode.children.push(node);
        }
      }
    }
    
    this.selectedNode = node;
    this.autoLayout();
    this.render();
    
    return node;
  }
  
  /**
   * Remove a factor from the graph
   */
  removeFactor(factorId) {
    const nodeIndex = this.nodes.findIndex(n => n.id === factorId);
    if (nodeIndex === -1) return;
    
    const node = this.nodes[nodeIndex];
    
    // Remove all edges connected to this node
    this.edges = this.edges.filter(edge => 
      edge.from.id !== factorId && edge.to.id !== factorId
    );
    
    // Remove from parent's children lists
    for (const potentialParent of this.nodes) {
      potentialParent.children = potentialParent.children.filter(
        child => child.id !== factorId
      );
    }
    
    // Remove the node
    this.nodes.splice(nodeIndex, 1);
    
    // If this was the selected node, deselect it
    if (this.selectedNode && this.selectedNode.id === factorId) {
      this.selectedNode = null;
    }
    
    this.render();
  }
  
  /**
   * Update a factor in the graph
   */
  updateFactor(factor) {
    const node = this.findNodeById(factor.name);
    if (!node) return this.addFactor(factor);
    
    // Update node properties
    node.label = factor.name;
    node.type = factor.factor_type;
    node.factor = factor;
    
    // Update edges based on new relationships
    // First, remove all edges coming to this node
    this.edges = this.edges.filter(edge => edge.to.id !== factor.name);
    
    // Remove this node from all parent children lists
    for (const potentialParent of this.nodes) {
      potentialParent.children = potentialParent.children.filter(
        child => child.id !== factor.name
      );
    }
    
    // Then, add new edges based on dependencies
    if (factor.relationships && factor.relationships.depends_on) {
      for (const parentId of factor.relationships.depends_on) {
        const parentNode = this.findNodeById(parentId);
        if (parentNode) {
          this.edges.push({
            from: parentNode,
            to: node
          });
          
          // Add to parent's children list
          parentNode.children.push(node);
        }
      }
    }
    
    this.render();
    return node;
  }
  
  /**
   * Find a node by ID
   */
  findNodeById(id) {
    return this.nodes.find(node => node.id === id);
  }
  
  /**
   * Generate proper hierarchical relationships for all nodes
   */
  generateHierarchicalRelationships() {
    const factors = [];
    
    // Process nodes in topological order (parents before children)
    const rootNodes = this.nodes.filter(node => 
      !this.edges.some(edge => edge.to.id === node.id)
    );
    
    // Helper function to process a node and its descendants
    const processNode = (node) => {
      const factor = { ...node.factor };
      
      // Get parents from edges
      const parentIds = this.edges
        .filter(edge => edge.to.id === node.id)
        .map(edge => edge.from.id);
      
      // Update relationships
      factor.relationships = {
        depends_on: parentIds
      };
      
      factors.push(factor);
      
      // Process children
      for (const edge of this.edges) {
        if (edge.from.id === node.id) {
          processNode(edge.to);
        }
      }
    };
    
    // Start processing from root nodes
    for (const rootNode of rootNodes) {
      processNode(rootNode);
    }
    
    return factors;
  }
  
  /**
   * Auto-layout the graph
   */
  autoLayout() {
    // Find root nodes (nodes with no parents)
    const rootNodes = this.nodes.filter(node => 
      !this.edges.some(edge => edge.to.id === node.id)
    );
    
    // Position root nodes at the top
    const rootSpacing = this.canvas.width / (rootNodes.length + 1);
    rootNodes.forEach((node, index) => {
      node.x = rootSpacing * (index + 1);
      node.y = 50;
    });
    
    // Layout helper function for recursively positioning nodes
    const layoutNode = (node, level, index, totalSiblings) => {
      // Position children
      const children = this.nodes.filter(n => 
        this.edges.some(edge => edge.from.id === node.id && edge.to.id === n.id)
      );
      
      const childSpacing = this.canvas.width / (children.length + 1);
      const levelHeight = 120;
      
      children.forEach((child, childIndex) => {
        child.x = childSpacing * (childIndex + 1);
        child.y = 50 + (level + 1) * levelHeight;
        
        // Layout grandchildren
        layoutNode(child, level + 1, childIndex, children.length);
      });
    };
    
    // Start layout from root nodes
    rootNodes.forEach((node, index) => {
      layoutNode(node, 0, index, rootNodes.length);
    });
    
    this.render();
  }
  
  /**
   * Render the graph
   */
  render() {
    const { ctx, canvas } = this;
    if (!ctx || !canvas) {
      console.error('Cannot render graph: canvas context not available');
      return;
    }
    
    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw a subtle grid for visual reference
    this.drawGrid();
    
    // Draw edges
    if (this.edges.length > 0) {
      ctx.lineWidth = 2;
      for (const edge of this.edges) {
        // Skip invalid edges
        if (!edge.from || !edge.to) {
          console.warn('Invalid edge', edge);
          continue;
        }
        
        // Calculate arrow points
        const fromX = edge.from.x;
        const fromY = edge.from.y;
        const toX = edge.to.x;
        const toY = edge.to.y;
        
        // Calculate direction vector
        const dx = toX - fromX;
        const dy = toY - fromY;
        const length = Math.sqrt(dx * dx + dy * dy);
        
        // Skip if nodes are at the same position
        if (length < 1) continue;
        
        const unitX = dx / length;
        const unitY = dy / length;
        
        // Calculate start and end points (adjusted for node radius)
        const startX = fromX + unitX * edge.from.radius;
        const startY = fromY + unitY * edge.from.radius;
        const endX = toX - unitX * edge.to.radius;
        const endY = toY - unitY * edge.to.radius;
        
        // Draw line
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.strokeStyle = '#64748b';
        ctx.stroke();
        
        // Draw arrow head
        const arrowLength = 10;
        const arrowWidth = 6;
        
        // Calculate arrow points
        const arrowX1 = endX - arrowLength * unitX + arrowWidth * unitY;
        const arrowY1 = endY - arrowLength * unitY - arrowWidth * unitX;
        const arrowX2 = endX - arrowLength * unitX - arrowWidth * unitY;
        const arrowY2 = endY - arrowLength * unitY + arrowWidth * unitX;
        
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(arrowX1, arrowY1);
        ctx.lineTo(arrowX2, arrowY2);
        ctx.closePath();
        ctx.fillStyle = '#64748b';
        ctx.fill();
      }
    } else if (this.nodes.length > 0) {
      // If we have nodes but no edges, show helper text
      ctx.font = '14px Inter, sans-serif';
      ctx.fillStyle = '#64748b';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Create relationships by selecting a node and clicking "Add Factor"', canvas.width / 2, 30);
    }
    
    // Draw nodes
    if (this.nodes.length > 0) {
      for (const node of this.nodes) {
        // Node shadow for depth
        ctx.beginPath();
        ctx.arc(node.x + 2, node.y + 2, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
        ctx.fill();
        
        // Node circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        
        // Fill based on factor type
        ctx.fillStyle = this.factorTypes[node.type] || '#64748b';
        ctx.fill();
        
        // Highlight selected node with thicker border
        if (this.selectedNode === node) {
          ctx.lineWidth = 4;
          ctx.strokeStyle = '#ffffff';
          ctx.stroke();
          
          // Draw a "child indicator" button
          ctx.beginPath();
          ctx.arc(node.x + node.radius * 0.7, node.y - node.radius * 0.7, 10, 0, Math.PI * 2);
          ctx.fillStyle = '#10b981';
          ctx.fill();
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 14px Inter, sans-serif';
          ctx.fillText('+', node.x + node.radius * 0.7, node.y - node.radius * 0.7 + 1);
        }
        
        // Node text
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.label, node.x, node.y);
        
        // Draw small labels for node type and dependencies
        ctx.font = '10px Inter, sans-serif';
        ctx.fillStyle = '#ffffff';
        const deps = node.factor.relationships.depends_on.length;
        const typeText = node.type.charAt(0).toUpperCase();
        const depText = deps > 0 ? `↑${deps}` : '';
        ctx.fillText(`${typeText}${depText}`, node.x, node.y + 14);
      }
    } else {
      // If no nodes, show help text
      ctx.font = '16px Inter, sans-serif';
      ctx.fillStyle = '#64748b';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Click "Add Factor" to create your first factor', canvas.width / 2, canvas.height / 2);
    }
    
    // Show factor count
    ctx.font = '12px Inter, sans-serif';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`Factors: ${this.nodes.length}`, 10, 10);
    
    // Show relationship count
    ctx.fillText(`Relationships: ${this.edges.length}`, 10, 30);
  }
  
  /**
   * Draw a subtle grid for visual reference
   */
  drawGrid() {
    const { ctx, canvas } = this;
    const gridSize = 40;
    
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.1)';
    ctx.lineWidth = 1;
    
    // Draw vertical lines
    for (let x = 0; x <= canvas.width; x += gridSize) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
    }
    
    // Draw horizontal lines
    for (let y = 0; y <= canvas.height; y += gridSize) {
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
    }
    
    ctx.stroke();
  }
  
  /**
   * Get all the factors in the graph with proper relationships
   */
  getFactors() {
    return this.generateHierarchicalRelationships();
  }
  
  /**
   * Clear the graph
   */
  clear() {
    this.nodes = [];
    this.edges = [];
    this.selectedNode = null;
    console.log('Cleared factor graph');
    this.render();
  }
  
  /**
   * Set graph data from existing factors
   */
  setFactors(factors) {
    console.log('Setting factors in graph:', factors);
    this.clear();
    
    if (!factors || factors.length === 0) {
      console.log('No factors to display');
      this.render();
      return;
    }
    
    // First pass: add all nodes without edges
    for (const factor of factors) {
      const node = {
        id: factor.name,
        label: factor.name,
        type: factor.factor_type,
        x: Math.random() * this.canvas.width * 0.8 + this.canvas.width * 0.1,
        y: Math.random() * this.canvas.height * 0.8 + this.canvas.height * 0.1,
        radius: 30,
        factor: factor,
        children: []
      };
      
      this.nodes.push(node);
    }
    
    // Second pass: add edges
    for (const factor of factors) {
      if (factor.relationships && 
          factor.relationships.depends_on && 
          factor.relationships.depends_on.length > 0) {
        
        const childNode = this.findNodeById(factor.name);
        
        for (const parentId of factor.relationships.depends_on) {
          const parentNode = this.findNodeById(parentId);
          
          if (parentNode && childNode) {
            console.log(`Adding edge from ${parentNode.id} to ${childNode.id}`);
            this.edges.push({
              from: parentNode,
              to: childNode
            });
            
            // Add to parent's children list
            parentNode.children.push(childNode);
          } else {
            console.warn(`Cannot create edge: parent=${parentId} or child=${factor.name} not found`);
          }
        }
      }
    }
    
    // Apply auto layout
    this.autoLayout();
    this.render();
  }
  
  /**
   * Quick add a factor directly to the graph
   */
  quickAddFactor() {
    // Generate unique name with timestamp to guarantee uniqueness 
    const factorCount = this.nodes.length;
    const timestamp = new Date().getTime().toString().slice(-4);
    const factorName = `factor_${factorCount + 1}_${timestamp}`;
    
    // Create a basic continuous factor
    const factor = {
      name: factorName,
      factor_type: 'continuous',
      value: 0,
      params: {
        loc: 0,
        scale: 1
      },
      relationships: {
        depends_on: []
      }
    };
    
    // If a node is selected, make this a child node
    if (this.selectedNode) {
      factor.relationships.depends_on = [this.selectedNode.id];
    }
    
    // Calculate position - put it somewhere visible but slightly random
    const x = this.canvas.width / 2 + Math.random() * 200 - 100;
    const y = this.canvas.height / 2 + Math.random() * 200 - 100;
    
    // Add to graph WITHOUT triggering edit
    const node = this.addFactor(factor, x, y);
    
    // Select the node but DO NOT open editor automatically
    this.selectedNode = node;
    this.render();
    
    return node;
  }
}

// Export for use in other modules
window.FactorGraph = FactorGraph; 