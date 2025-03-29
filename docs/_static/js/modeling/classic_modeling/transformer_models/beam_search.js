
// Auto-converted from TSX using Babel - Browser Compatible Version
(function() {
    // Get React from global scope
    var React = window.React;

    // Make React hooks available
    var useState = React.useState;
    var useEffect = React.useEffect;
    var useRef = React.useRef;

    // Simple JSX runtime replacement
    var jsx = function(type, props, key, children) {
        var newProps = props || {};
        if (children !== undefined) {
            newProps.children = children;
        }
        return React.createElement(type, newProps, children);
    };

    var jsxs = function(type, props, key, children) {
        return jsx(type, props, key, children);
    };

    // Helper functions from Babel
    function _extends() {
        _extends = Object.assign || function(target) {
            for (var i = 1; i < arguments.length; i++) {
                var source = arguments[i];
                for (var key in source) {
                    if (Object.prototype.hasOwnProperty.call(source, key)) {
                        target[key] = source[key];
                    }
                }
            }
            return target;
        };
        return _extends.apply(this, arguments);
    }

    // Babel transpiled component with export statements removed
function _typeof(o) { "@babel/helpers - typeof"; return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (o) { return typeof o; } : function (o) { return o && "function" == typeof Symbol && o.constructor === Symbol && o !== Symbol.prototype ? "symbol" : typeof o; }, _typeof(o); }
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == _typeof(i) ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != _typeof(t) || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != _typeof(i)) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
function _toConsumableArray(r) { return _arrayWithoutHoles(r) || _iterableToArray(r) || _unsupportedIterableToArray(r) || _nonIterableSpread(); }
function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }
function _iterableToArray(r) { if ("undefined" != typeof Symbol && null != r[Symbol.iterator] || null != r["@@iterator"]) return Array.from(r); }
function _arrayWithoutHoles(r) { if (Array.isArray(r)) return _arrayLikeToArray(r); }
function _slicedToArray(r, e) { return _arrayWithHoles(r) || _iterableToArrayLimit(r, e) || _unsupportedIterableToArray(r, e) || _nonIterableRest(); }
function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }
function _unsupportedIterableToArray(r, a) { if (r) { if ("string" == typeof r) return _arrayLikeToArray(r, a); var t = {}.toString.call(r).slice(8, -1); return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray(r, a) : void 0; } }
function _arrayLikeToArray(r, a) { (null == a || a > r.length) && (a = r.length); for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e]; return n; }
function _iterableToArrayLimit(r, l) { var t = null == r ? null : "undefined" != typeof Symbol && r[Symbol.iterator] || r["@@iterator"]; if (null != t) { var e, n, i, u, a = [], f = !0, o = !1; try { if (i = (t = t.call(r)).next, 0 === l) { if (Object(t) !== t) return; f = !1; } else for (; !(f = (e = i.call(t)).done) && (a.push(e.value), a.length !== l); f = !0); } catch (r) { o = !0, n = r; } finally { try { if (!f && null != t.return && (u = t.return(), Object(u) !== u)) return; } finally { if (o) throw n; } } return a; } }
function _arrayWithHoles(r) { if (Array.isArray(r)) return r; }
// Using global React object (window.React) instead of imports

var BeamSearchVisualization = function BeamSearchVisualization() {
  var _useState = useState(2),
    _useState2 = _slicedToArray(_useState, 2),
    beamWidth = _useState2[0],
    setBeamWidth = _useState2[1];
  var _useState3 = useState(5),
    _useState4 = _slicedToArray(_useState3, 2),
    currentStep = _useState4[0],
    setCurrentStep = _useState4[1]; // Start at the final step, now we have 6 steps (0-5)
  var _useState5 = useState(5),
    _useState6 = _slicedToArray(_useState5, 2),
    maxStep = _useState6[0],
    setMaxStep = _useState6[1];
  var _useState7 = useState([]),
    _useState8 = _slicedToArray(_useState7, 2),
    nodes = _useState8[0],
    setNodes = _useState8[1];
  var _useState9 = useState([]),
    _useState10 = _slicedToArray(_useState9, 2),
    edges = _useState10[0],
    setEdges = _useState10[1];
  var _useState11 = useState(false),
    _useState12 = _slicedToArray(_useState11, 2),
    autoPlay = _useState12[0],
    setAutoPlay = _useState12[1];
  var _useState13 = useState(1500),
    _useState14 = _slicedToArray(_useState13, 2),
    autoPlaySpeed = _useState14[0],
    setAutoPlaySpeed = _useState14[1];
  var _useState15 = useState(0.6),
    _useState16 = _slicedToArray(_useState15, 2),
    lengthPenalty = _useState16[0],
    setLengthPenalty = _useState16[1]; // Default length penalty alpha
  var _useState17 = useState(1000),
    _useState18 = _slicedToArray(_useState17, 2),
    containerWidth = _useState18[0],
    setContainerWidth = _useState18[1]; // Wider default for 6 columns
  var containerRef = useRef(null);

  // Track container size for responsive layout
  useEffect(function () {
    var updateWidth = function updateWidth() {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.offsetWidth);
      }
    };

    // Initial measurement
    updateWidth();

    // Add resize event listener
    window.addEventListener('resize', updateWidth);
    return function () {
      return window.removeEventListener('resize', updateWidth);
    };
  }, []);

  // Setup initial data
  useEffect(function () {
    resetVisualization();
  }, [beamWidth, lengthPenalty]);

  // Auto-play functionality
  useEffect(function () {
    var timer;
    if (autoPlay && currentStep < maxStep) {
      timer = setTimeout(function () {
        setCurrentStep(function (prev) {
          return Math.min(prev + 1, maxStep);
        });
      }, autoPlaySpeed);
    }
    return function () {
      return clearTimeout(timer);
    };
  }, [autoPlay, currentStep, maxStep, autoPlaySpeed]);

  // Function to apply length normalization
  var normalizeScoreByLength = function normalizeScoreByLength(score, length, alpha) {
    if (alpha === 0) return score; // No normalization

    // Standard length penalty formula from GNMT paper:
    // score / (((5 + length)^alpha) / ((5 + 1)^alpha))
    var lengthPenalty = Math.pow(5 + length, alpha) / Math.pow(6, alpha);
    return score / lengthPenalty;
  };

  // Generate beam search data based on beam width
  var generateBeamSearchData = function generateBeamSearchData(width) {
    var nodes = [];
    var edges = [];

    // Predefined set of probabilities to make visualization more realistic
    var scoreOptions = [-0.3, -0.4, -0.6, -0.7, -0.8, -0.9, -1.1, -1.2, -1.5];

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
    var firstTokens = ['A', 'B', 'C', 'D', 'E'];
    var firstTokenScores = [{
      token: 'A',
      score: -0.6
    }, {
      token: 'B',
      score: -1.2
    }, {
      token: 'C',
      score: -0.3
    }, {
      token: 'D',
      score: -0.4
    }, {
      token: 'E',
      score: -1.5
    }];
    firstTokenScores.forEach(function (item) {
      var id = item.token;
      var length = 1; // Single token
      var normalizedScore = normalizeScoreByLength(item.score, length, lengthPenalty);
      nodes.push({
        id: id,
        text: item.token,
        score: item.score,
        cumulative: item.score,
        normalizedScore: normalizedScore,
        isSelected: false,
        // Will update after sorting
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
    var step1Candidates = nodes.filter(function (n) {
      return n.step === 1;
    }).map(function (n) {
      return {
        id: n.id,
        normalizedScore: n.normalizedScore
      };
    });
    var step1Selected = step1Candidates.sort(function (a, b) {
      return a.normalizedScore - b.normalizedScore;
    }).slice(0, width).map(function (item) {
      return item.id;
    });

    // Update selected status
    nodes.forEach(function (node) {
      if (node.step === 1) {
        node.isSelected = step1Selected.includes(node.id);
      }
    });

    // Step 3: Generate second tokens for each selected first token - First Repeat Expansion
    var selectedStep1Nodes = nodes.filter(function (n) {
      return n.step === 1 && n.isSelected;
    });
    var step2Candidates = [];
    selectedStep1Nodes.forEach(function (parentNode) {
      // Each selected token from step 1 generates second tokens
      firstTokens.forEach(function (token) {
        var score = scoreOptions[Math.floor(Math.random() * scoreOptions.length)];
        var id = "".concat(parentNode.text).concat(token);
        var cumulative = parentNode.cumulative + score;
        var length = parentNode.length + 1;
        var normalizedScore = normalizeScoreByLength(cumulative, length, lengthPenalty);
        nodes.push({
          id: id,
          text: id,
          score: score,
          cumulative: cumulative,
          normalizedScore: normalizedScore,
          isSelected: false,
          // Will update after sorting
          isEos: false,
          isFinalOutput: false,
          step: 2,
          parentId: parentNode.id,
          length: length
        });
        edges.push({
          source: parentNode.id,
          target: id,
          score: score
        });
        step2Candidates.push({
          id: id,
          cumulative: cumulative,
          normalizedScore: normalizedScore,
          parentId: parentNode.id
        });
      });
    });

    // Step 4: Select top-k for second tokens - First Repeat Selection, based on normalized scores
    var step2Selected = step2Candidates.sort(function (a, b) {
      return a.normalizedScore - b.normalizedScore;
    }).slice(0, width).map(function (item) {
      return item.id;
    });

    // Update selected status
    nodes.forEach(function (node) {
      if (node.step === 2) {
        node.isSelected = step2Selected.includes(node.id);
      }
    });

    // Step 5: Generate third tokens for each selected second token - Second Repeat Expansion
    var selectedStep2Nodes = nodes.filter(function (n) {
      return n.step === 2 && n.isSelected;
    });
    var step3Candidates = [];

    // Make one EOS token at step 3 for the first parent's first option
    // but ensure other sequences continue
    selectedStep2Nodes.forEach(function (parentNode, parentIndex) {
      // Generate third tokens
      firstTokens.slice(0, 3).forEach(function (token, tokenIndex) {
        // First parent gets an EOS token for its first option
        var isEos = parentIndex === 0 && tokenIndex === 0;

        // For EOS, use a score that's not the best (to ensure the beam continues)
        var score = isEos ? -0.9 : scoreOptions[Math.floor(Math.random() * scoreOptions.length)];
        var id = isEos ? "".concat(parentNode.text, "-EOS") : "".concat(parentNode.text).concat(token);
        var cumulative = parentNode.cumulative + score;
        var length = parentNode.length + 1;
        var normalizedScore = normalizeScoreByLength(cumulative, length, lengthPenalty);
        nodes.push({
          id: id,
          text: isEos ? "".concat(parentNode.text, "<EOS>") : "".concat(parentNode.text).concat(token),
          score: score,
          cumulative: cumulative,
          normalizedScore: normalizedScore,
          isSelected: false,
          isEos: isEos,
          isFinalOutput: false,
          step: 3,
          parentId: parentNode.id,
          length: length
        });
        edges.push({
          source: parentNode.id,
          target: id,
          score: score
        });
        step3Candidates.push({
          id: id,
          cumulative: cumulative,
          normalizedScore: normalizedScore,
          isEos: isEos,
          parentId: parentNode.id
        });
      });
    });

    // Step 6: Select top-k for third tokens - Second Repeat Selection, based on normalized scores
    // Ensure we keep some non-EOS tokens to continue to the final step
    var eosNodes = step3Candidates.filter(function (c) {
      return c.isEos;
    });
    var nonEosNodes = step3Candidates.filter(function (c) {
      return !c.isEos;
    });

    // Sort by normalized score
    eosNodes.sort(function (a, b) {
      return a.normalizedScore - b.normalizedScore;
    });
    nonEosNodes.sort(function (a, b) {
      return a.normalizedScore - b.normalizedScore;
    });

    // Select the best nodes, making sure at least one non-EOS continues
    var step3Selected = [];

    // If beam width is 1, use the single best node
    if (width === 1) {
      var allNodesSorted = [].concat(_toConsumableArray(eosNodes), _toConsumableArray(nonEosNodes)).sort(function (a, b) {
        return a.normalizedScore - b.normalizedScore;
      });
      step3Selected = [allNodesSorted[0].id];
    } else {
      // For beam width > 1, ensure we keep at least one EOS and one non-EOS
      if (eosNodes.length > 0) {
        step3Selected.push(eosNodes[0].id);
        step3Selected = [].concat(_toConsumableArray(step3Selected), _toConsumableArray(nonEosNodes.slice(0, width - 1).map(function (n) {
          return n.id;
        })));
      } else {
        step3Selected = nonEosNodes.slice(0, width).map(function (n) {
          return n.id;
        });
      }
    }

    // Update selected status
    nodes.forEach(function (node) {
      if (node.step === 3) {
        node.isSelected = step3Selected.includes(node.id);
      }
    });

    // Step 7: Generate fourth tokens for selected third tokens that aren't EOS - Final Expansion
    var selectedStep3Nodes = nodes.filter(function (n) {
      return n.step === 3 && n.isSelected && !n.isEos;
    });
    var step4Candidates = [];

    // Generate more tokens for the continuing beams
    selectedStep3Nodes.forEach(function (parentNode) {
      // Generate final tokens, with one being EOS
      firstTokens.slice(0, 2).forEach(function (token, tokenIndex) {
        // Make all tokens EOS for simplicity in the final step
        var isEos = tokenIndex === 0; // First token becomes EOS

        var score = scoreOptions[Math.floor(Math.random() * scoreOptions.length)];
        var id = isEos ? "".concat(parentNode.text, "-EOS") : "".concat(parentNode.text).concat(token);
        var cumulative = parentNode.cumulative + score;
        var length = parentNode.length + 1;
        var normalizedScore = normalizeScoreByLength(cumulative, length, lengthPenalty);
        nodes.push({
          id: id,
          text: isEos ? "".concat(parentNode.text, "<EOS>") : "".concat(parentNode.text).concat(token),
          score: score,
          cumulative: cumulative,
          normalizedScore: normalizedScore,
          isSelected: false,
          // Will update after final selection
          isEos: isEos,
          isFinalOutput: false,
          step: 4,
          parentId: parentNode.id,
          length: length
        });
        edges.push({
          source: parentNode.id,
          target: id,
          score: score
        });
        step4Candidates.push({
          id: id,
          cumulative: cumulative,
          normalizedScore: normalizedScore,
          isEos: isEos,
          parentId: parentNode.id
        });
      });
    });

    // Step 8: Final selection - choose the best EOS token as the final output
    // Combine EOS tokens from step 3 and step 4
    var allEosNodes = [].concat(_toConsumableArray(eosNodes), _toConsumableArray(step4Candidates.filter(function (n) {
      return n.isEos;
    })));

    // Sort by normalized score to find the best EOS token
    allEosNodes.sort(function (a, b) {
      return a.normalizedScore - b.normalizedScore;
    });

    // Select the best EOS node from all steps
    if (allEosNodes.length > 0) {
      var bestEosId = allEosNodes[0].id;

      // Mark the best EOS node as the final output
      nodes.forEach(function (node) {
        if (node.id === bestEosId) {
          node.isFinalOutput = true;

          // If this is a step 4 node, mark it as selected to show the path
          if (node.step === 4) {
            node.isSelected = true;

            // Also mark any best non-EOS tokens as selected
            var nonEosStep4 = step4Candidates.filter(function (n) {
              return !n.isEos;
            }).sort(function (a, b) {
              return a.normalizedScore - b.normalizedScore;
            }).slice(0, width - 1);
            nonEosStep4.forEach(function (bestNonEos) {
              var nonEosNode = nodes.find(function (n) {
                return n.id === bestNonEos.id;
              });
              if (nonEosNode) nonEosNode.isSelected = true;
            });
          }
        }
      });
    }

    // For the final output step (step 5), create a special node that connects to the best EOS node
    // Create a "Final Result" node at step 5
    var bestOutput = nodes.find(function (n) {
      return n.isFinalOutput;
    });
    if (bestOutput) {
      var finalNodeId = "output-final";
      var finalNode = {
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

      // Connect it to the best EOS node
      edges.push({
        source: bestOutput.id,
        target: finalNodeId,
        score: 0
      });
    }
    return {
      nodes: nodes,
      edges: edges
    };
  };
  var resetVisualization = function resetVisualization() {
    var _generateBeamSearchDa = generateBeamSearchData(beamWidth),
      nodes = _generateBeamSearchDa.nodes,
      edges = _generateBeamSearchDa.edges;
    setNodes(nodes);
    setEdges(edges);
    setCurrentStep(5); // Start showing the complete visualization
  };

  // Filter nodes and edges based on current step
  var visibleNodes = nodes.filter(function (node) {
    return node.step <= currentStep;
  });
  var visibleEdges = edges.filter(function (edge) {
    var sourceNode = nodes.find(function (n) {
      return n.id === edge.source;
    });
    var targetNode = nodes.find(function (n) {
      return n.id === edge.target;
    });
    return sourceNode && targetNode && sourceNode.step <= currentStep && targetNode.step <= currentStep;
  });

  // Calculate responsive column positions with more horizontal space
  var calculateColumnPositions = function calculateColumnPositions(width) {
    var padding = 40; // Smaller padding for more columns
    var usableWidth = width - padding * 2;
    // Use a wider spacing for better separation between columns
    var columnCount = 6;
    return [{
      x: padding + usableWidth * 0 / 5,
      title: "Initialization",
      subtitle: "Start",
      step: 0
    }, {
      x: padding + usableWidth * 1 / 5,
      title: "Expansion",
      subtitle: "First Tokens",
      step: 1
    }, {
      x: padding + usableWidth * 2 / 5,
      title: "Selection",
      subtitle: "Keep Top-K",
      step: 2
    }, {
      x: padding + usableWidth * 3 / 5,
      title: "Expansion (Repeat)",
      subtitle: "Next Tokens",
      step: 3
    }, {
      x: padding + usableWidth * 4 / 5,
      title: "Selection (Repeat)",
      subtitle: "With EOS Tokens",
      step: 4
    }, {
      x: padding + usableWidth * 5 / 5,
      title: "Final Output",
      subtitle: "Best Sequence",
      step: 5
    }];
  };

  // Responsive column positions
  var columnPositions = calculateColumnPositions(containerWidth);

  // Organize nodes by parent for hierarchical layout with better spacing
  var organizeByHierarchy = function organizeByHierarchy() {
    // Group nodes by step
    var nodesByStep = [visibleNodes.filter(function (n) {
      return n.step === 0;
    }), visibleNodes.filter(function (n) {
      return n.step === 1;
    }), visibleNodes.filter(function (n) {
      return n.step === 2;
    }), visibleNodes.filter(function (n) {
      return n.step === 3;
    }), visibleNodes.filter(function (n) {
      return n.step === 4;
    }), visibleNodes.filter(function (n) {
      return n.step === 5;
    })];

    // Calculate positions
    var positionedNodes = _toConsumableArray(visibleNodes).map(function (node) {
      return _objectSpread(_objectSpread({}, node), {}, {
        renderX: columnPositions[node.step].x,
        renderY: 0 // Will be updated later
      });
    });
    nodesByStep.forEach(function (stepNodes, stepIndex) {
      // We want to evenly space the nodes
      var totalNodes = stepNodes.length;
      var startY = 180; // Top margin for nodes (extra space for subtitles)
      var gapBetweenNodes = totalNodes > 1 ? 80 : 0; // Increased vertical gap between nodes

      // Group nodes by parent for step 2 and later
      if (stepIndex >= 2) {
        var nodesByParent = {};

        // Group by parent
        stepNodes.forEach(function (node) {
          if (node.parentId) {
            if (!nodesByParent[node.parentId]) {
              nodesByParent[node.parentId] = [];
            }
            nodesByParent[node.parentId].push(node);
          }
        });

        // Get parent nodes
        var parentStepNodes = nodesByStep[stepIndex - 1].filter(function (n) {
          return n.isSelected;
        }).sort(function (a, b) {
          var _positionedNodes$find, _positionedNodes$find2;
          // Find the position of these nodes
          var posA = ((_positionedNodes$find = positionedNodes.find(function (p) {
            return p.id === a.id;
          })) === null || _positionedNodes$find === void 0 ? void 0 : _positionedNodes$find.renderY) || 0;
          var posB = ((_positionedNodes$find2 = positionedNodes.find(function (p) {
            return p.id === b.id;
          })) === null || _positionedNodes$find2 === void 0 ? void 0 : _positionedNodes$find2.renderY) || 0;
          return posA - posB;
        });

        // Position children under their parents
        parentStepNodes.forEach(function (parent) {
          var children = nodesByParent[parent.id] || [];
          var parentPos = positionedNodes.find(function (p) {
            return p.id === parent.id;
          });
          if (parentPos) {
            // Place children around the parent's Y position
            var parentY = parentPos.renderY;
            var totalChildren = children.length;
            if (totalChildren > 0) {
              var childSpacing = 70; // Increased spacing between child nodes
              var startChildY = parentY - (totalChildren - 1) * childSpacing / 2;
              children.forEach(function (child, childIndex) {
                var node = positionedNodes.find(function (n) {
                  return n.id === child.id;
                });
                if (node) {
                  node.renderY = startChildY + childIndex * childSpacing;
                }
              });
            }
          }
        });
      } else {
        // For step 0 and 1, just space evenly
        stepNodes.forEach(function (node, index) {
          var posNode = positionedNodes.find(function (n) {
            return n.id === node.id;
          });
          if (posNode) {
            posNode.renderY = startY + index * gapBetweenNodes;
          }
        });
      }
    });
    return positionedNodes.filter(function (n) {
      return n.renderX !== undefined && n.renderY !== undefined;
    });
  };

  // Calculate node positions with hierarchical layout
  var positionedNodes = organizeByHierarchy();

  // Calculate SVG viewBox for responsiveness
  var svgWidth = containerWidth;
  var svgHeight = 750;
  var viewBox = "0 0 ".concat(svgWidth, " ").concat(svgHeight);
  return /*#__PURE__*/React.createElement("div", {
    className: "flex flex-col w-full h-full"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between items-center mb-4 p-4 bg-gray-100 rounded-lg"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center space-x-4"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "mr-2 font-bold"
  }, "Beam Width:"), /*#__PURE__*/React.createElement("select", {
    value: beamWidth,
    onChange: function onChange(e) {
      return setBeamWidth(parseInt(e.target.value));
    },
    className: "p-2 border rounded bg-white text-lg"
  }, [1, 2, 3, 4, 5].map(function (width) {
    return /*#__PURE__*/React.createElement("option", {
      key: width,
      value: width
    }, width);
  }))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "mr-2 font-bold"
  }, "Length Penalty (\u03B1):"), /*#__PURE__*/React.createElement("select", {
    value: lengthPenalty,
    onChange: function onChange(e) {
      return setLengthPenalty(parseFloat(e.target.value));
    },
    className: "p-2 border rounded bg-white text-lg"
  }, /*#__PURE__*/React.createElement("option", {
    value: "0"
  }, "0 (No Penalty)"), /*#__PURE__*/React.createElement("option", {
    value: "0.6"
  }, "0.6 (Standard)"), /*#__PURE__*/React.createElement("option", {
    value: "1.0"
  }, "1.0 (Linear)"), /*#__PURE__*/React.createElement("option", {
    value: "1.5"
  }, "1.5 (Strong)")))), /*#__PURE__*/React.createElement("div", {
    className: "flex space-x-2"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: function onClick() {
      return setCurrentStep(function (prev) {
        return Math.max(prev - 1, 0);
      });
    },
    disabled: currentStep === 0,
    className: "px-4 py-2 bg-blue-500 text-white text-lg rounded disabled:bg-gray-300"
  }, "Previous"), /*#__PURE__*/React.createElement("button", {
    onClick: function onClick() {
      return setCurrentStep(function (prev) {
        return Math.min(prev + 1, maxStep);
      });
    },
    disabled: currentStep === maxStep,
    className: "px-4 py-2 bg-blue-500 text-white text-lg rounded disabled:bg-gray-300"
  }, "Next"), /*#__PURE__*/React.createElement("button", {
    onClick: function onClick() {
      return setAutoPlay(!autoPlay);
    },
    className: "px-4 py-2 text-white text-lg rounded ".concat(autoPlay ? 'bg-red-500' : 'bg-green-500')
  }, autoPlay ? 'Pause' : 'Play'), /*#__PURE__*/React.createElement("button", {
    onClick: resetVisualization,
    className: "px-4 py-2 bg-gray-500 text-white text-lg rounded"
  }, "Reset"))), /*#__PURE__*/React.createElement("div", {
    ref: containerRef,
    className: "relative border rounded-lg bg-white overflow-hidden",
    style: {
      height: '800px'
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: "100%",
    height: svgHeight,
    viewBox: viewBox,
    preserveAspectRatio: "xMidYMid meet"
  }, columnPositions.map(function (column, idx) {
    return /*#__PURE__*/React.createElement("g", {
      key: "header-".concat(idx)
    }, /*#__PURE__*/React.createElement("text", {
      x: column.x,
      y: "30",
      fontSize: "18",
      fontWeight: "bold",
      textAnchor: "middle",
      fill: "#000"
    }, column.title), /*#__PURE__*/React.createElement("text", {
      x: column.x,
      y: "55",
      fontSize: "14",
      textAnchor: "middle",
      fill: "#666"
    }, "Step ", column.step), /*#__PURE__*/React.createElement("text", {
      x: column.x,
      y: "80",
      fontSize: "13",
      fontStyle: "italic",
      textAnchor: "middle",
      fill: "#666"
    }, column.subtitle));
  }), columnPositions.slice(0, -1).map(function (column, idx) {
    return /*#__PURE__*/React.createElement("line", {
      key: "separator-".concat(idx),
      x1: (column.x + columnPositions[idx + 1].x) / 2,
      y1: "110",
      x2: (column.x + columnPositions[idx + 1].x) / 2,
      y2: "700",
      stroke: "#e5e7eb",
      strokeWidth: "1",
      strokeDasharray: "4,4"
    });
  }), visibleEdges.map(function (edge) {
    var sourceNode = positionedNodes.find(function (n) {
      return n.id === edge.source;
    });
    var targetNode = positionedNodes.find(function (n) {
      return n.id === edge.target;
    });
    if (!sourceNode || !targetNode) return null;
    var isSelected = sourceNode.isSelected && targetNode.isSelected;
    var isFinalPath = targetNode.isFinalOutput || sourceNode.isFinalOutput;
    return /*#__PURE__*/React.createElement("g", {
      key: "edge-".concat(edge.source, "-").concat(edge.target)
    }, /*#__PURE__*/React.createElement("line", {
      x1: sourceNode.renderX,
      y1: sourceNode.renderY,
      x2: targetNode.renderX,
      y2: targetNode.renderY,
      stroke: isFinalPath ? "#1e8449" : isSelected ? "#3182ce" : "#a0aec0",
      strokeWidth: isFinalPath ? 3 : isSelected ? 2.5 : 1.5,
      strokeDasharray: isSelected ? "5,3" : ""
    }), /*#__PURE__*/React.createElement("text", {
      x: (sourceNode.renderX + targetNode.renderX) / 2,
      y: (sourceNode.renderY + targetNode.renderY) / 2 - 10,
      fontSize: "14",
      textAnchor: "middle",
      fill: "#4a5568"
    }, edge.score.toFixed(1)));
  }), positionedNodes.map(function (node) {
    return /*#__PURE__*/React.createElement("g", {
      key: "node-".concat(node.id),
      transform: "translate(".concat(node.renderX, ", ").concat(node.renderY, ")")
    }, /*#__PURE__*/React.createElement("circle", {
      r: 22,
      fill: node.isFinalOutput ? "#d5f5e3" : node.isEos ? "#f5cba7" : node.isSelected ? "#aed6f1" : "#f0f0f0",
      stroke: node.isFinalOutput ? "#1e8449" : node.isEos ? "#e67e22" : node.isSelected ? "#2874a6" : "#000",
      strokeWidth: node.isSelected || node.isFinalOutput || node.isEos ? 3 : 2
    }), /*#__PURE__*/React.createElement("text", {
      textAnchor: "middle",
      dominantBaseline: "middle",
      fontSize: node.isEos ? 12 : 15 // Smaller font for EOS token
      ,
      fontWeight: node.isSelected || node.isFinalOutput ? "bold" : "normal"
    }, node.text), (currentStep >= 1 || node.step > 0) && node.step < 5 && /*#__PURE__*/React.createElement("g", null, /*#__PURE__*/React.createElement("text", {
      textAnchor: "middle",
      dominantBaseline: "middle",
      fontSize: "12",
      y: 30,
      fill: "#4a5568"
    }, "raw: ", node.cumulative.toFixed(1)), /*#__PURE__*/React.createElement("text", {
      textAnchor: "middle",
      dominantBaseline: "middle",
      fontSize: "12",
      y: 44,
      fill: "#4a5568",
      fontWeight: "medium"
    }, "norm: ", node.normalizedScore.toFixed(2))), node.step === 5 && /*#__PURE__*/React.createElement("g", null, /*#__PURE__*/React.createElement("text", {
      textAnchor: "middle",
      dominantBaseline: "middle",
      fontSize: "12",
      y: 30,
      fill: "#1e8449"
    }, "raw: ", node.cumulative.toFixed(1)), /*#__PURE__*/React.createElement("text", {
      textAnchor: "middle",
      dominantBaseline: "middle",
      fontSize: "12",
      y: 44,
      fill: "#1e8449",
      fontWeight: "bold"
    }, "normalized: ", node.normalizedScore.toFixed(2))));
  }), /*#__PURE__*/React.createElement("g", {
    transform: "translate(70, 680)"
  }, /*#__PURE__*/React.createElement("rect", {
    x: "-65",
    y: "-25",
    width: "450",
    height: "85",
    rx: "4",
    fill: "#f8f9fa",
    stroke: "#e2e8f0",
    strokeWidth: "1"
  }), /*#__PURE__*/React.createElement("text", {
    x: "0",
    y: "0",
    fontSize: "15",
    fontWeight: "bold",
    fill: "#2d3748"
  }, "Length Penalty Formula:"), /*#__PURE__*/React.createElement("text", {
    x: "0",
    y: "25",
    fontSize: "14",
    fill: "#4a5568",
    fontFamily: "monospace"
  }, "score / (((5 + length)^", lengthPenalty, ") / ((5 + 1)^", lengthPenalty, "))"), /*#__PURE__*/React.createElement("text", {
    x: "0",
    y: "50",
    fontSize: "13",
    fill: "#4a5568",
    fontStyle: "italic"
  }, "\u03B1 = ", lengthPenalty, " ", lengthPenalty === 0 ? "(favors shorter sequences)" : lengthPenalty < 1 ? "(mildly encourages length)" : "(strongly encourages length)")), /*#__PURE__*/React.createElement("g", {
    transform: "translate(650, 680)"
  }, /*#__PURE__*/React.createElement("circle", {
    cx: "0",
    cy: "0",
    r: "12",
    fill: "#aed6f1",
    stroke: "#2874a6",
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("text", {
    x: "20",
    y: "5",
    fontSize: "14"
  }, "Selected Beam"), /*#__PURE__*/React.createElement("circle", {
    cx: "0",
    cy: "30",
    r: "12",
    fill: "#f5cba7",
    stroke: "#e67e22",
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("text", {
    x: "20",
    y: "35",
    fontSize: "14"
  }, "EOS Token"), /*#__PURE__*/React.createElement("rect", {
    x: "-12",
    y: "50",
    width: "24",
    height: "24",
    rx: "5",
    fill: "#d5f5e3",
    stroke: "#1e8449",
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("text", {
    x: "20",
    y: "65",
    fontSize: "14"
  }, "Final Output")), /*#__PURE__*/React.createElement("g", null, /*#__PURE__*/React.createElement("text", {
    x: "20",
    y: "730",
    fontSize: "14",
    fill: "#4a5568"
  }, "Beam Width: ", beamWidth), /*#__PURE__*/React.createElement("text", {
    x: svgWidth / 2,
    y: "730",
    fontSize: "14",
    textAnchor: "middle",
    fill: "#4a5568"
  }, "Step: ", currentStep, " / ", maxStep), /*#__PURE__*/React.createElement("text", {
    x: svgWidth - 20,
    y: "730",
    fontSize: "14",
    textAnchor: "end",
    fill: "#4a5568"
  }, "Values shown are log probabilities")))), /*#__PURE__*/React.createElement("div", {
    className: "mt-4 p-4 bg-gray-100 rounded-lg"
  }, /*#__PURE__*/React.createElement("h3", {
    className: "text-xl font-bold mb-3"
  }, "How Beam Search Works:"), /*#__PURE__*/React.createElement("ol", {
    className: "list-decimal list-inside space-y-2 text-lg"
  }, /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("strong", null, "Initialization:"), " Start with an empty sequence"), /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("strong", null, "Expansion:"), " For each beam, calculate scores for all possible next tokens"), /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("strong", null, "Selection:"), " Keep only top-", beamWidth, " highest scoring sequences"), /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("strong", null, "Repeat:"), " Continue expanding and selecting until EOS token or max length", /*#__PURE__*/React.createElement("ul", {
    className: "list-disc ml-8 mt-1 text-base"
  }, /*#__PURE__*/React.createElement("li", null, "When an EOS token is encountered, that sequence is complete"), /*#__PURE__*/React.createElement("li", null, "Other beams continue expanding until they also reach EOS or max length"))), /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("strong", null, "Output:"), " Return highest scoring complete sequence (with EOS token)")), /*#__PURE__*/React.createElement("div", {
    className: "mt-4 p-3 bg-white rounded border border-gray-300"
  }, /*#__PURE__*/React.createElement("h4", {
    className: "font-bold text-lg mb-2"
  }, "Length Normalization"), /*#__PURE__*/React.createElement("p", {
    className: "mb-2"
  }, "Without length normalization, beam search tends to favor shorter sequences because each additional token can only reduce the overall probability. Length normalization helps counteract this bias."), /*#__PURE__*/React.createElement("div", {
    className: "mb-2"
  }, /*#__PURE__*/React.createElement("strong", null, "Formula:"), " score / (((5 + length)^\u03B1) / ((5 + 1)^\u03B1))"), /*#__PURE__*/React.createElement("ul", {
    className: "list-disc ml-8 mb-2"
  }, /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("strong", null, "\u03B1 = 0:"), " No normalization (favors shorter sequences)"), /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("strong", null, "\u03B1 = 0.6:"), " Standard value that mildly encourages longer sequences"), /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("strong", null, "\u03B1 = 1.0 or higher:"), " Strongly encourages longer sequences")), /*#__PURE__*/React.createElement("p", null, "This visualization shows both raw scores (cumulative log probabilities) and normalized scores that account for sequence length."))));
};

/* export removed */ /* BeamSearchVisualization will be assigned to window.BeamSearchVisualization */
// Global export
if (typeof window !== 'undefined') {
  window.beam_search = beam_search;
}


    // Make sure the component is exported to the global window object
    if (typeof window !== 'undefined' && typeof beam_search !== 'undefined') {
        window.beam_search = beam_search;
        console.log('Successfully exported beam_search to global scope');
    }
})();
