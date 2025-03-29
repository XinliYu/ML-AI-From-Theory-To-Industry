import React, { useState, useEffect, useRef } from 'react';

type NodeType = {
  id: string;
  text: string;
  score: number;
  cumulative: number; // Raw cumulative score
  normalizedScore: number; // Length-normalized score
  isSelected: boolean;
  isEos: boolean;
  isFinalOutput: boolean;
  step: number;
  parentId?: string;
  length: number; // Token length for normalization
};

type EdgeType = {
  source: string;
  target: string;
  score: number;
};

const BeamSearchVisualization = () => {
  const [beamWidth, setBeamWidth] = useState(2);
  const [currentStep, setCurrentStep] = useState(5); // Start at the final step, now we have 6 steps (0-5)
  const [maxStep, setMaxStep] = useState(5);
  const [nodes, setNodes] = useState<NodeType[]>([]);
  const [edges, setEdges] = useState<EdgeType[]>([]);
  const [autoPlay, setAutoPlay] = useState(false);
  const [autoPlaySpeed, setAutoPlaySpeed] = useState(1500);
  const [lengthPenalty, setLengthPenalty] = useState(0.6); // Default length penalty alpha
  const [containerWidth, setContainerWidth] = useState(1300); // Wider default for 6 columns
  const containerRef = useRef<HTMLDivElement>(null);

  // Track container size for responsive layout
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.offsetWidth);
      }
    };

    // Initial measurement
    updateWidth();

    // Add resize event listener
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  // Setup initial data
  useEffect(() => {
    resetVisualization();
  }, [beamWidth, lengthPenalty]);

  // Auto-play functionality
  useEffect(() => {
    let timer: NodeJS.Timeout;

    if (autoPlay && currentStep < maxStep) {
      timer = setTimeout(() => {
        setCurrentStep(prev => Math.min(prev + 1, maxStep));
      }, autoPlaySpeed);
    }

    return () => clearTimeout(timer);
  }, [autoPlay, currentStep, maxStep, autoPlaySpeed]);

  // Add MathJax for rendering LaTeX formulas
  useEffect(() => {
    // Only add the script once
    if (!window.MathJax) {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js';
      script.async = true;
      script.onload = () => {
        // Process math on the page once loaded
        if (window.MathJax && window.MathJax.typeset) {
          window.MathJax.typeset();
        }
      };
      document.head.appendChild(script);
    } else {
      // If MathJax is already loaded, just reprocess
      if (window.MathJax && window.MathJax.typeset) {
        window.MathJax.typeset();
      }
    }
  }, [lengthPenalty]); // Re-typeset when length penalty changes

  // Function to apply length normalization
  const normalizeScoreByLength = (score: number, length: number, alpha: number): number => {
    if (alpha === 0) return score; // No normalization

    // Standard length penalty formula from GNMT paper:
    // score / (((5 + length)^alpha) / ((5 + 1)^alpha))
    const lengthPenalty = Math.pow(5 + length, alpha) / Math.pow(6, alpha);
    return score / lengthPenalty;
  };

  // Generate beam search data based on beam width
  const generateBeamSearchData = (width: number): {nodes: NodeType[], edges: EdgeType[]} => {
    const nodes: NodeType[] = [];
    const edges: EdgeType[] = [];

    // Predefined set of probabilities to make visualization more realistic
    const scoreOptions = [-0.3, -0.4, -0.6, -0.7, -0.8, -0.9, -1.1, -1.2, -1.5];

    // Initial start node
    nodes.push({
      id: 'start',
      text: '[]',
      score: 0,
      cumulative: 0,
      normalizedScore: 0,
      isSelected: true,
      isEos: false,
      isFinalOutput: false,
      step: 0,
      length: 0
    });

    // Step 1: First tokens (A, B, C, D, E) - Initial Expansion
    const firstTokens = ['A', 'B', 'C', 'D', 'E'];
    const firstTokenScores = [
      { token: 'A', score: -0.6 },
      { token: 'B', score: -1.2 },
      { token: 'C', score: -0.3 },
      { token: 'D', score: -0.4 },
      { token: 'E', score: -1.5 }
    ];

    firstTokenScores.forEach((item) => {
      const id = item.token;
      const length = 1; // Single token
      const normalizedScore = normalizeScoreByLength(item.score, length, lengthPenalty);

      nodes.push({
        id,
        text: item.token,
        score: item.score,
        cumulative: item.score,
        normalizedScore: normalizedScore,
        isSelected: false, // Will update after sorting
        isEos: false,
        isFinalOutput: false,
        step: 1,
        parentId: 'start',
        length: length
      });

      edges.push({
        source: 'start',
        target: id,
        score: item.score
      });
    });

    // Step 2: Select top-k for first tokens - Initial Selection, based on normalized scores
    const step1Candidates = nodes.filter(n => n.step === 1)
      .map(n => ({ id: n.id, normalizedScore: n.normalizedScore }));

    const step1Selected = step1Candidates
      .sort((a, b) => a.normalizedScore - b.normalizedScore)
      .slice(0, width)
      .map(item => item.id);

    // Update selected status
    nodes.forEach(node => {
      if (node.step === 1) {
        node.isSelected = step1Selected.includes(node.id);
      }
    });

    // Step 3: Generate second tokens for each selected first token - First Repeat Expansion
    const selectedStep1Nodes = nodes.filter(n => n.step === 1 && n.isSelected);
    let step2Candidates: {id: string, cumulative: number, normalizedScore: number, parentId: string}[] = [];

    selectedStep1Nodes.forEach(parentNode => {
      // Each selected token from step 1 generates second tokens
      firstTokens.forEach(token => {
        const score = scoreOptions[Math.floor(Math.random() * scoreOptions.length)];
        const id = `${parentNode.text}${token}`;
        const cumulative = parentNode.cumulative + score;
        const length = parentNode.length + 1;
        const normalizedScore = normalizeScoreByLength(cumulative, length, lengthPenalty);

        nodes.push({
          id,
          text: id,
          score,
          cumulative,
          normalizedScore,
          isSelected: false, // Will update after sorting
          isEos: false,
          isFinalOutput: false,
          step: 2,
          parentId: parentNode.id,
          length
        });

        edges.push({
          source: parentNode.id,
          target: id,
          score
        });

        step2Candidates.push({ id, cumulative, normalizedScore, parentId: parentNode.id });
      });
    });

    // Step 4: Select top-k for second tokens - First Repeat Selection, based on normalized scores
    const step2Selected = step2Candidates
      .sort((a, b) => a.normalizedScore - b.normalizedScore)
      .slice(0, width)
      .map(item => item.id);

    // Update selected status
    nodes.forEach(node => {
      if (node.step === 2) {
        node.isSelected = step2Selected.includes(node.id);
      }
    });

    // Step 5: Generate third tokens for each selected second token - Second Repeat Expansion
    const selectedStep2Nodes = nodes.filter(n => n.step === 2 && n.isSelected);
    let step3Candidates: {id: string, cumulative: number, normalizedScore: number, isEos: boolean, parentId: string}[] = [];

    // Make one EOS token at step 3 for the first parent's first option
    // but ensure other sequences continue
    selectedStep2Nodes.forEach((parentNode, parentIndex) => {
      // Generate third tokens
      firstTokens.slice(0, 3).forEach((token, tokenIndex) => {
        // First parent gets an EOS token for its first option
        const isEos = parentIndex === 0 && tokenIndex === 0;

        // For EOS, use a score that's not the best (to ensure the beam continues)
        const score = isEos ? -0.9 : scoreOptions[Math.floor(Math.random() * scoreOptions.length)];
        const id = isEos ? `${parentNode.text}-EOS` : `${parentNode.text}${token}`;
        const cumulative = parentNode.cumulative + score;
        const length = parentNode.length + 1;
        const normalizedScore = normalizeScoreByLength(cumulative, length, lengthPenalty);

        nodes.push({
          id,
          text: isEos ? `${parentNode.text}<EOS>` : `${parentNode.text}${token}`,
          score,
          cumulative,
          normalizedScore,
          isSelected: false,
          isEos,
          isFinalOutput: false,
          step: 3,
          parentId: parentNode.id,
          length
        });

        edges.push({
          source: parentNode.id,
          target: id,
          score
        });

        step3Candidates.push({ id, cumulative, normalizedScore, isEos, parentId: parentNode.id });
      });
    });

    // Step 6: Select top-k for third tokens - Second Repeat Selection, based on normalized scores
    // Ensure we keep some non-EOS tokens to continue to the final step
    const eosNodes = step3Candidates.filter(c => c.isEos);
    const nonEosNodes = step3Candidates.filter(c => !c.isEos);

    // Sort by normalized score
    eosNodes.sort((a, b) => a.normalizedScore - b.normalizedScore);
    nonEosNodes.sort((a, b) => a.normalizedScore - b.normalizedScore);

    // Select the best nodes, making sure at least one non-EOS continues
    let step3Selected: string[] = [];

    // If beam width is 1, use the single best node
    if (width === 1) {
      const allNodesSorted = [...eosNodes, ...nonEosNodes].sort((a, b) => a.normalizedScore - b.normalizedScore);
      step3Selected = [allNodesSorted[0].id];
    } else {
      // For beam width > 1, ensure we keep at least one EOS and one non-EOS
      if (eosNodes.length > 0) {
        step3Selected.push(eosNodes[0].id);
        step3Selected = [...step3Selected, ...nonEosNodes.slice(0, width - 1).map(n => n.id)];
      } else {
        step3Selected = nonEosNodes.slice(0, width).map(n => n.id);
      }
    }

    // Update selected status
    nodes.forEach(node => {
      if (node.step === 3) {
        node.isSelected = step3Selected.includes(node.id);
      }
    });

    // Step 7: Generate fourth tokens for selected third tokens that aren't EOS - Final Expansion
    const selectedStep3Nodes = nodes.filter(n => n.step === 3 && n.isSelected && !n.isEos);
    let step4Candidates: {id: string, cumulative: number, normalizedScore: number, isEos: boolean, parentId: string}[] = [];

    // Generate more tokens for the continuing beams
    selectedStep3Nodes.forEach((parentNode) => {
      // Generate final tokens, with one being EOS
      firstTokens.slice(0, 2).forEach((token, tokenIndex) => {
        // Make all tokens EOS for simplicity in the final step
        const isEos = tokenIndex === 0; // First token becomes EOS

        const score = scoreOptions[Math.floor(Math.random() * scoreOptions.length)];
        const id = isEos ? `${parentNode.text}-EOS` : `${parentNode.text}${token}`;
        const cumulative = parentNode.cumulative + score;
        const length = parentNode.length + 1;
        const normalizedScore = normalizeScoreByLength(cumulative, length, lengthPenalty);

        nodes.push({
          id,
          text: isEos ? `${parentNode.text}<EOS>` : `${parentNode.text}${token}`,
          score,
          cumulative,
          normalizedScore,
          isSelected: false, // Will update after final selection
          isEos,
          isFinalOutput: false,
          step: 4,
          parentId: parentNode.id,
          length
        });

        edges.push({
          source: parentNode.id,
          target: id,
          score
        });

        step4Candidates.push({ id, cumulative, normalizedScore, isEos, parentId: parentNode.id });
      });
    });

    // Step 8: Final selection - choose the best EOS token as the final output
    // Combine EOS tokens from step 3 and step 4
    const allEosNodes = [...eosNodes, ...step4Candidates.filter(n => n.isEos)];

    // Sort by normalized score to find the best EOS token
    allEosNodes.sort((a, b) => b.normalizedScore - a.normalizedScore);

    // Select the best EOS node from all steps
    if (allEosNodes.length > 0) {
      const bestEosId = allEosNodes[0].id;

      // Mark the best EOS node as the final output
      nodes.forEach(node => {
        if (node.id === bestEosId) {
          node.isFinalOutput = true;

          // If this is a step 4 node, mark it as selected to show the path
          if (node.step === 4) {
            node.isSelected = true;

            // Also mark any best non-EOS tokens as selected
            const nonEosStep4 = step4Candidates
              .filter(n => !n.isEos)
              .sort((a, b) => a.normalizedScore - b.normalizedScore)
              .slice(0, width - 1);

            nonEosStep4.forEach(bestNonEos => {
              const nonEosNode = nodes.find(n => n.id === bestNonEos.id);
              if (nonEosNode) nonEosNode.isSelected = true;
            });
          }
        }
      });
    }

    // For the final output step (step 5), create a special node that connects to the best EOS node
    // Create a "Final Result" node at step 5
    const bestOutput = nodes.find(n => n.isFinalOutput);

    if (bestOutput) {
      const finalNodeId = "output-final";

      // Find the position of the best output node
      const bestOutputStep = bestOutput.step;

      // Create final node that will be placed horizontally aligned
      const finalNode: NodeType = {
        id: finalNodeId,
        text: bestOutput.text,
        score: 0,
        cumulative: bestOutput.cumulative,
        normalizedScore: bestOutput.normalizedScore,
        isSelected: true,
        isEos: true,
        isFinalOutput: true,
        step: 5,
        parentId: bestOutput.id,
        length: bestOutput.length
      };

      nodes.push(finalNode);

      // Connect it to the best EOS node with a score of 0
      edges.push({
        source: bestOutput.id,
        target: finalNodeId,
        score: 0
      });
    }

    return { nodes, edges };
  };

  const resetVisualization = () => {
    const { nodes, edges } = generateBeamSearchData(beamWidth);
    setNodes(nodes);
    setEdges(edges);
    setCurrentStep(5); // Start showing the complete visualization
  };

  // Filter nodes and edges based on current step
  const visibleNodes = nodes.filter(node => node.step <= currentStep);
  const visibleEdges = edges.filter(edge => {
    const sourceNode = nodes.find(n => n.id === edge.source);
    const targetNode = nodes.find(n => n.id === edge.target);
    return sourceNode && targetNode &&
           sourceNode.step <= currentStep &&
           targetNode.step <= currentStep;
  });

  // Calculate responsive column positions with more horizontal space
  const calculateColumnPositions = (width: number) => {
    const padding = 40; // Smaller padding for more columns
    const usableWidth = width - (padding * 2);
    // Use a wider spacing for better separation between columns
    const columnCount = 6;

    return [
      { x: padding + (usableWidth * 0/5), title: "Init", subtitle: "Start", step: 0 },
      { x: padding + (usableWidth * 1/5), title: "Expansion", subtitle: "First Tokens", step: 1 },
      { x: padding + (usableWidth * 2/5), title: "Selection", subtitle: "Keep Top-K", step: 2 },
      { x: padding + (usableWidth * 3/5), title: "Expansion (Repeat)", subtitle: "Next Tokens", step: 3 },
      { x: padding + (usableWidth * 4/5), title: "Selection (Repeat)", subtitle: "Keep Top-K", step: 4 },
      { x: padding + (usableWidth * 5/5), title: "Output", subtitle: "Best Sequence", step: 5 }
    ];
  };

  // Responsive column positions
  const columnPositions = calculateColumnPositions(containerWidth);

  // Organize nodes by parent for hierarchical layout with better spacing
  const organizeByHierarchy = () => {
    // Group nodes by step
    const nodesByStep = [
      visibleNodes.filter(n => n.step === 0),
      visibleNodes.filter(n => n.step === 1),
      visibleNodes.filter(n => n.step === 2),
      visibleNodes.filter(n => n.step === 3),
      visibleNodes.filter(n => n.step === 4),
      visibleNodes.filter(n => n.step === 5)
    ];

    // Calculate positions
    const positionedNodes = [...visibleNodes].map(node => ({
      ...node,
      renderX: columnPositions[node.step].x,
      renderY: 0 // Will be updated later
    }));

    nodesByStep.forEach((stepNodes, stepIndex) => {
      // We want to evenly space the nodes
      const totalNodes = stepNodes.length;
      const startY = 220; // Top margin for nodes (extra space for subtitles)
      const gapBetweenNodes = totalNodes > 1 ? 80 : 0; // Increased vertical gap between nodes

      // Group nodes by parent for step 2 and later
      if (stepIndex >= 2) {
        const nodesByParent: Record<string, NodeType[]> = {};

        // Group by parent
        stepNodes.forEach(node => {
          if (node.parentId) {
            if (!nodesByParent[node.parentId]) {
              nodesByParent[node.parentId] = [];
            }
            nodesByParent[node.parentId].push(node);
          }
        });

        // Get parent nodes
        const parentStepNodes = nodesByStep[stepIndex - 1]
          .filter(n => n.isSelected)
          .sort((a, b) => {
            // Find the position of these nodes
            const posA = positionedNodes.find(p => p.id === a.id)?.renderY || 0;
            const posB = positionedNodes.find(p => p.id === b.id)?.renderY || 0;
            return posA - posB;
          });

        // Position children under their parents
        parentStepNodes.forEach(parent => {
          const children = nodesByParent[parent.id] || [];
          const parentPos = positionedNodes.find(p => p.id === parent.id);

          if (parentPos) {
            // Place children around the parent's Y position
            const parentY = parentPos.renderY;
            const totalChildren = children.length;

            if (totalChildren > 0) {
              const childSpacing = 70; // Increased spacing between child nodes
              const startChildY = parentY - ((totalChildren - 1) * childSpacing) / 2;

              children.forEach((child, childIndex) => {
                const node = positionedNodes.find(n => n.id === child.id);
                if (node) {
                  node.renderY = startChildY + (childIndex * childSpacing);
                }
              });
            }
          }
        });
      }

      // Special handling for step 5 (final output) nodes
      if (stepIndex === 5) {
        stepNodes.forEach(node => {
          // Find its parent node
          const parentNode = positionedNodes.find(n => n.id === node.parentId);
          if (parentNode) {
            // Place the final output node at the same Y position as its parent
            // This ensures the horizontal line connecting them works correctly
            const finalNodePos = positionedNodes.find(n => n.id === node.id);
            if (finalNodePos) {
              finalNodePos.renderY = parentNode.renderY;
            }
          } else {
            // Fallback positioning if parent not found
            const node5 = positionedNodes.find(n => n.id === node.id);
            if (node5) {
              node5.renderY = startY;
            }
          }
        });
      } else if (stepIndex < 2) {
        // For step 0 and 1, just space evenly
        stepNodes.forEach((node, index) => {
          const posNode = positionedNodes.find(n => n.id === node.id);
          if (posNode) {
            posNode.renderY = startY + (index * gapBetweenNodes);
          }
        });
      }
    });

    return positionedNodes.filter(n => n.renderX !== undefined && n.renderY !== undefined);
  };

  // Calculate node positions with hierarchical layout
  const positionedNodes = organizeByHierarchy();

  // Function to find empty space for legend
  const findLegendPosition = () => {
    // Define potential positions to try (in order of preference)
    const positions = [
      { x: svgWidth - 190, y: 120 },     // Top right (original)
      { x: svgWidth - 190, y: 500 },     // Middle right
      { x: svgWidth - 190, y: 650 },     // Bottom right
      { x: 20, y: 120 },                 // Top left
      { x: 20, y: 650 },                 // Bottom left
      { x: svgWidth / 2 - 85, y: 120 },  // Top center
      { x: svgWidth / 2 - 85, y: 650 }   // Bottom center
    ];

    // Legend dimensions
    const legendWidth = 170;
    const legendHeight = 110;

    // Find first position with no node collisions
    for (const pos of positions) {
      let hasCollision = false;

      // Check if any visible node overlaps with this position
      for (const node of positionedNodes) {
        if (!node.renderX || !node.renderY) continue;

        // Simple bounding box collision detection
        const legendLeft = pos.x;
        const legendRight = pos.x + legendWidth;
        const legendTop = pos.y;
        const legendBottom = pos.y + legendHeight;

        // Node bounding box (with padding for text)
        const nodeLeft = node.renderX - 55;
        const nodeRight = node.renderX + 55;
        const nodeTop = node.renderY - 55;
        const nodeBottom = node.renderY + 55;

        // Check collision
        if (!(nodeLeft > legendRight ||
              nodeRight < legendLeft ||
              nodeTop > legendBottom ||
              nodeBottom < legendTop)) {
          hasCollision = true;
          break;
        }
      }

      // If no collision, use this position
      if (!hasCollision) {
        return pos;
      }
    }

    // Fallback to original position if all have collisions
    return positions[0];
  };

  // Get the best position
  const legendPos = findLegendPosition();

  // Calculate SVG viewBox for responsiveness
  const svgWidth = containerWidth;
  const svgHeight = 950;
  const viewBox = `0 0 ${svgWidth} ${svgHeight}`;

  return (
    <div className="flex flex-col w-full h-full">
      <div className="flex justify-between items-center mb-4 p-4 bg-gray-100 rounded-lg">
        <div className="flex items-center space-x-4">
          <div>
            <span className="mr-2 font-bold">Beam Width:</span>
            <select
              value={beamWidth}
              onChange={(e) => setBeamWidth(parseInt(e.target.value))}
              className="p-2 border rounded bg-white text-lg"
            >
              {[1, 2, 3, 4, 5].map(width => (
                <option key={width} value={width}>{width}</option>
              ))}
            </select>
          </div>

          <div>
            <span className="mr-2 font-bold">Length Penalty (α):</span>
            <select
              value={lengthPenalty}
              onChange={(e) => setLengthPenalty(parseFloat(e.target.value))}
              className="p-2 border rounded bg-white text-lg"
            >
              <option value="0">0 (No Penalty)</option>
              <option value="0.6">0.6 (Standard)</option>
              <option value="1.0">1.0 (Linear)</option>
              <option value="1.5">1.5 (Strong)</option>
            </select>
          </div>
        </div>

        <div className="flex space-x-2">
          <button
            onClick={() => setCurrentStep(prev => Math.max(prev - 1, 0))}
            disabled={currentStep === 0}
            className="px-4 py-2 bg-blue-500 text-white text-lg rounded disabled:bg-gray-300"
          >
            Previous
          </button>

          <button
            onClick={() => setCurrentStep(prev => Math.min(prev + 1, maxStep))}
            disabled={currentStep === maxStep}
            className="px-4 py-2 bg-blue-500 text-white text-lg rounded disabled:bg-gray-300"
          >
            Next
          </button>

          <button
            onClick={() => setAutoPlay(!autoPlay)}
            className={`px-4 py-2 text-white text-lg rounded ${autoPlay ? 'bg-red-500' : 'bg-green-500'}`}
          >
            {autoPlay ? 'Pause' : 'Play'}
          </button>

          <button
            onClick={resetVisualization}
            className="px-4 py-2 bg-gray-500 text-white text-lg rounded"
          >
            Reset
          </button>
        </div>
      </div>

      <div ref={containerRef} className="relative border rounded-lg bg-white overflow-hidden" style={{ height: '800px' }}>
        {/* Headers are rendered inside the SVG for better alignment */}
        <svg width="100%" height={svgHeight} viewBox={viewBox} preserveAspectRatio="xMidYMid meet">
          {/* Step titles */}
          {columnPositions.map((column, idx) => (
              <g key={`header-${idx}`}>
                <text
                    x={column.x}
                    y="30"
                    fontSize="18"
                    fontWeight="bold"
                    textAnchor="middle"
                    fill="#000"
                >
                  {column.title}
                </text>
                <text
                    x={column.x}
                    y="55"
                    fontSize="14"
                    textAnchor="middle"
                    fill="#666"
                >
                  Step {column.step}
                </text>
                <text
                    x={column.x}
                    y="80"
                    fontSize="13"
                    fontStyle="italic"
                    textAnchor="middle"
                    fill="#666"
                >
                  {column.subtitle}
                </text>
              </g>
          ))}

          {/* Vertical separation lines */}
          {columnPositions.slice(0, -1).map((column, idx) => (
              <line
                  key={`separator-${idx}`}
                  x1={(column.x + columnPositions[idx + 1].x) / 2}
                  y1="110"
                  x2={(column.x + columnPositions[idx + 1].x) / 2}
                  y2="700"
                  stroke="#e5e7eb"
                  strokeWidth="1"
                  strokeDasharray="4,4"
              />
          ))}

          {/* Edges */}
          {visibleEdges.map(edge => {
            const sourceNode = positionedNodes.find(n => n.id === edge.source);
            const targetNode = positionedNodes.find(n => n.id === edge.target);

            if (!sourceNode || !targetNode) return null;

            const isSelected = sourceNode.isSelected && targetNode.isSelected;
            const isFinalPath = targetNode.isFinalOutput || sourceNode.isFinalOutput;

            // Special case for final output edge - make it horizontal
            if (targetNode.step === 5 && targetNode.isFinalOutput) {
              return (
                  <g key={`edge-${edge.source}-${edge.target}`}>
                    <line
                        x1={sourceNode.renderX}
                        y1={sourceNode.renderY}
                        x2={targetNode.renderX}
                        y2={sourceNode.renderY} // Use source Y to keep it horizontal
                        stroke="#1e8449"
                        strokeWidth={3}
                    />
                    <text
                        x={(sourceNode.renderX + targetNode.renderX) / 2}
                        y={sourceNode.renderY - 10}
                        fontSize="14"
                        textAnchor="middle"
                        fill="#4a5568"
                    >
                      {edge.score.toFixed(1)}
                    </text>
                  </g>
              );
            }

            return (
                // Original edge code
                <g key={`edge-${edge.source}-${edge.target}`}>
                  <line
                      x1={sourceNode.renderX}
                      y1={sourceNode.renderY}
                      x2={targetNode.renderX}
                      y2={targetNode.renderY}
                      stroke={isFinalPath ? "#1e8449" : isSelected ? "#3182ce" : "#a0aec0"}
                      strokeWidth={isFinalPath ? 3 : isSelected ? 2.5 : 1.5}
                      strokeDasharray={isSelected ? "5,3" : ""}
                  />
                  <text
                      x={(sourceNode.renderX + targetNode.renderX) / 2}
                      y={((sourceNode.renderY + targetNode.renderY) / 2) - 10}
                      fontSize="14"
                      textAnchor="middle"
                      fill="#4a5568"
                  >
                    {edge.score.toFixed(1)}
                  </text>
                </g>
            );
          })}

          {/* Nodes */}
          {positionedNodes.map(node => (
              <g key={`node-${node.id}`} transform={`translate(${node.renderX}, ${node.renderY})`}>
                <circle
                    r={22}
                    fill={node.isFinalOutput ? "#d5f5e3" : node.isEos ? "#f5cba7" : node.isSelected ? "#aed6f1" : "#f0f0f0"}
                    stroke={node.isFinalOutput ? "#1e8449" : node.isEos ? "#e67e22" : node.isSelected ? "#2874a6" : "#000"}
                    strokeWidth={node.isSelected || node.isFinalOutput || node.isEos ? 3 : 2}
                />
                <text
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize={node.isEos ? 12 : 15}  // Smaller font for EOS token
                    fontWeight={node.isSelected || node.isFinalOutput ? "bold" : "normal"}
                >
                  {node.text}
                </text>
                {(currentStep >= 1 || node.step > 0) && node.step < 5 && (
                    <g>
                      <text
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize="12"
                          y={30}
                          fill="#4a5568"
                      >
                        raw: {node.cumulative.toFixed(1)}
                      </text>
                      <text
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize="12"
                          y={44}
                          fill="#4a5568"
                          fontWeight="medium"
                      >
                        norm: {node.normalizedScore.toFixed(2)}
                      </text>
                    </g>
                )}
                {node.step === 5 && (
                    <g>
                      <text
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize="12"
                          y={30}
                          fill="#1e8449"
                      >
                        raw: {node.cumulative.toFixed(1)}
                      </text>
                      <text
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize="12"
                          y={44}
                          fill="#1e8449"
                          fontWeight="bold"
                      >
                        normalized: {node.normalizedScore.toFixed(2)}
                      </text>
                    </g>
                )}
              </g>
          ))}

          {/* Legend */}
          <g>
            <rect x={legendPos.x} y={legendPos.y} width="170" height="110" rx="5"
                  fill="#f8f9fa" stroke="#e2e8f0" strokeWidth="1" opacity="0.9"/>

            <circle cx={legendPos.x + 30} cy={legendPos.y + 25} r="12"
                    fill="#aed6f1" stroke="#2874a6" strokeWidth="2"/>
            <text x={legendPos.x + 55} y={legendPos.y + 30} fontSize="14">Selected Beam</text>

            <circle cx={legendPos.x + 30} cy={legendPos.y + 55} r="12"
                    fill="#f5cba7" stroke="#e67e22" strokeWidth="2"/>
            <text x={legendPos.x + 55} y={legendPos.y + 60} fontSize="14">EOS Token</text>

            <rect x={legendPos.x + 18} y={legendPos.y + 85} width="24" height="24" rx="5"
                  fill="#d5f5e3" stroke="#1e8449" strokeWidth="2"/>
            <text x={legendPos.x + 55} y={legendPos.y + 100} fontSize="14">Output</text>
          </g>

          {/* Status information */}
          <g>
            <text x="20" y="780" fontSize="14" fill="#4a5568">Beam Width: {beamWidth}</text>
            <text x={svgWidth / 2} y="780" fontSize="14" textAnchor="middle"
                  fill="#4a5568">Step: {currentStep} / {maxStep}</text>
            <text x={svgWidth - 20} y="780" fontSize="14" textAnchor="end" fill="#4a5568">Values shown are log
              probabilities
            </text>
          </g>
        </svg>
      </div>

      <div className="mt-8 p-4 bg-gray-100 rounded-lg">
        <h3 className="text-xl font-bold mb-3">How Beam Search Works:</h3>
        <ol className="list-decimal list-inside space-y-2 text-lg">
          <li><strong>Initialization:</strong> Start with an empty sequence</li>
          <li><strong>Expansion:</strong> For each beam, calculate scores for all possible next tokens</li>
          <li><strong>Selection:</strong> Keep only top-{beamWidth} highest scoring sequences</li>
          <li><strong>Repeat:</strong> Continue expanding and selecting until EOS token or max length
            <ul className="list-disc ml-8 mt-1 text-base">
              <li>When an EOS token is encountered, that sequence is complete</li>
              <li>Other beams continue expanding until they also reach EOS or max length</li>
            </ul>
          </li>
          <li><strong>Output:</strong> Return highest scoring complete sequence (with EOS token)</li>
        </ol>

        <div className="mt-4 p-3 bg-white rounded border border-gray-300">
          <p>This visualization shows both raw scores (cumulative log probabilities) and normalized scores that account
            for sequence length.</p>
        </div>
        <div className="mt-3 p-3 bg-gray-50 rounded">
          <p className="font-semibold mb-2">Length Penalty Formula:</p>
          <div className="flex justify-center my-4">
            <div id="length-penalty-formula" className="text-lg">
              {'$$\\text{score} / (\\frac{5 + \\text{length}}{5 + 1})^{\\alpha}$$'}
            </div>
          </div>
          <p className="mt-2 text-sm text-gray-600 italic text-center">
            where α = {lengthPenalty} {lengthPenalty === 0 ? "(favors shorter sequences)" :
              lengthPenalty < 1 ? "(mildly encourages length)." :
              "(strongly encourages length)."}
          </p>
        </div>
      </div>
    </div>
  );
};

export default BeamSearchVisualization;