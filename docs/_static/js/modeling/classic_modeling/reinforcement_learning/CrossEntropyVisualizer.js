
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
// Using global React object (window.React) instead of imports

var CrossEntropyLossAndGradient = function CrossEntropyLossAndGradient() {
  var canvasRef = useRef(null);

  // Sigmoid function
  var sigmoid = function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  };

  // Binary cross-entropy loss function
  var binaryCrossEntropy = function binaryCrossEntropy(z, r) {
    var p = sigmoid(z);
    // Avoid numerical instability with a safer implementation
    if (r === 1) {
      // For r=1, use -log(p) with a more numerically stable approach for large negative z
      return z >= 0 ? Math.log(1 + Math.exp(-z)) : -z + Math.log(1 + Math.exp(z));
    } else {
      // For r=0, use -log(1-p) with a more numerically stable approach for large positive z
      return z >= 0 ? z + Math.log(1 + Math.exp(-z)) : Math.log(1 + Math.exp(z));
    }
  };
  useEffect(function () {
    var canvas = canvasRef.current;
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var width = canvas.width;
    var height = canvas.height;

    // Define padding and plot dimensions with more space between plots
    var padding = {
      left: 60,
      right: 200,
      top: 80,
      bottom: 50
    };
    var plotHeight = (height - padding.top - padding.bottom - 60) / 2; // More space between plots
    var plotWidth = width - padding.left - padding.right;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);

    // Define the range for the logit z
    var zMin = -6;
    var zMax = 6;
    var zRange = zMax - zMin;

    // Define the range for the loss (higher range to show more of the curve)
    var lossMin = 0;
    var lossMax = 8;
    var lossRange = lossMax - lossMin;

    // Define the range for the gradient
    var gradMin = -1.1;
    var gradMax = 1.1;
    var gradRange = gradMax - gradMin;

    // Scales for mapping data to pixels
    var xScale = plotWidth / zRange;
    var lossYScale = plotHeight / lossRange;
    var gradYScale = plotHeight / gradRange;

    // Location of second plot (gradient) with more space
    var secondPlotTop = padding.top + plotHeight + 60; // Increased space between plots

    // Draw title
    ctx.font = 'bold 18px Arial';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#333';
    ctx.fillText('Binary Cross-Entropy Loss and Gradient', width / 2, 30);

    // ============ Draw Loss Plot (Top) ============

    // Draw axes for loss plot
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();

    // Y-axis
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + plotHeight);

    // X-axis
    ctx.moveTo(padding.left, padding.top + plotHeight);
    ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
    ctx.stroke();

    // Draw grid lines and labels for loss plot
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 0.5;

    // X-axis grid lines and labels (shared between plots)
    var xTicks = [-6, -4, -2, 0, 2, 4, 6];
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.font = '12px Arial';
    ctx.fillStyle = '#333';
    xTicks.forEach(function (z) {
      var x = padding.left + (z - zMin) * xScale;

      // Grid line for loss plot
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + plotHeight);
      ctx.stroke();

      // Grid line for gradient plot
      ctx.beginPath();
      ctx.moveTo(x, secondPlotTop);
      ctx.lineTo(x, secondPlotTop + plotHeight);
      ctx.stroke();

      // Label (only on gradient plot x-axis)
      ctx.fillText(z.toString(), x, secondPlotTop + plotHeight + 10);
    });

    // Y-axis grid lines and labels for loss plot
    var lossTicks = [0, 2, 4, 6, 8];
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    lossTicks.forEach(function (loss) {
      var y = padding.top + plotHeight - (loss - lossMin) * lossYScale;

      // Grid line
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + plotWidth, y);
      ctx.stroke();

      // Label
      ctx.fillText(loss.toFixed(1), padding.left - 10, y);
    });

    // Draw loss curves
    // Curve for r=0: loss = -log(1-sigmoid(z))
    ctx.strokeStyle = '#0066cc';
    ctx.lineWidth = 2;
    ctx.beginPath();
    var lastValidY0 = null;
    for (var x = 0; x <= plotWidth; x += 1) {
      var z = zMin + x / plotWidth * zRange;
      var loss = binaryCrossEntropy(z, 0);
      if (isNaN(loss) || !isFinite(loss)) {
        loss = lossMax; // Cap at max value
      }
      var plotX = padding.left + x;
      var rawY = padding.top + plotHeight - (loss - lossMin) * lossYScale;
      var plotY = Math.max(padding.top, rawY); // Ensure we stay within plot bounds

      if (x === 0 || lastValidY0 === null) {
        ctx.moveTo(plotX, plotY);
        lastValidY0 = plotY;
      } else {
        // If the value jumps too much, draw a vertical line to indicate discontinuity
        if (Math.abs(plotY - lastValidY0) > 50 && plotY === padding.top) {
          ctx.lineTo(plotX, lastValidY0);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(plotX, plotY);
        } else {
          ctx.lineTo(plotX, plotY);
        }
        lastValidY0 = plotY;
      }
    }
    ctx.stroke();

    // Curve for r=1: loss = -log(sigmoid(z))
    ctx.strokeStyle = '#cc3300';
    ctx.lineWidth = 2;
    ctx.beginPath();
    var lastValidY1 = null;
    for (var _x = 0; _x <= plotWidth; _x += 1) {
      var _z = zMin + _x / plotWidth * zRange;
      var _loss = binaryCrossEntropy(_z, 1);
      if (isNaN(_loss) || !isFinite(_loss)) {
        _loss = lossMax; // Cap at max value
      }
      var _plotX = padding.left + _x;
      var _rawY = padding.top + plotHeight - (_loss - lossMin) * lossYScale;
      var _plotY = Math.max(padding.top, _rawY); // Ensure we stay within plot bounds

      if (_x === 0 || lastValidY1 === null) {
        ctx.moveTo(_plotX, _plotY);
        lastValidY1 = _plotY;
      } else {
        // If the value jumps too much, draw a vertical line to indicate discontinuity
        if (Math.abs(_plotY - lastValidY1) > 50 && _plotY === padding.top) {
          ctx.lineTo(_plotX, lastValidY1);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(_plotX, _plotY);
        } else {
          ctx.lineTo(_plotX, _plotY);
        }
        lastValidY1 = _plotY;
      }
    }
    ctx.stroke();

    // Add a visual indication that the curve continues upward
    // For r=0 at right side
    var rightEdgeX = padding.left + plotWidth;
    var rightEdgeY0 = padding.top + plotHeight - (binaryCrossEntropy(zMax, 0) - lossMin) * lossYScale;
    if (rightEdgeY0 <= padding.top + 20) {
      ctx.strokeStyle = '#0066cc';
      ctx.beginPath();
      ctx.moveTo(rightEdgeX - 10, padding.top + 10);
      ctx.lineTo(rightEdgeX, padding.top);
      ctx.lineTo(rightEdgeX - 10, padding.top - 10);
      ctx.stroke();
    }

    // For r=1 at left side
    var leftEdgeX = padding.left;
    var leftEdgeY1 = padding.top + plotHeight - (binaryCrossEntropy(zMin, 1) - lossMin) * lossYScale;
    if (leftEdgeY1 <= padding.top + 20) {
      ctx.strokeStyle = '#cc3300';
      ctx.beginPath();
      ctx.moveTo(leftEdgeX + 10, padding.top + 10);
      ctx.lineTo(leftEdgeX, padding.top);
      ctx.lineTo(leftEdgeX + 10, padding.top - 10);
      ctx.stroke();
    }

    // ============ Draw Gradient Plot (Bottom) ============

    // Draw axes for gradient plot
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();

    // Y-axis
    ctx.moveTo(padding.left, secondPlotTop);
    ctx.lineTo(padding.left, secondPlotTop + plotHeight);

    // X-axis (at y=0)
    var gradYAxisZero = secondPlotTop + plotHeight / 2;
    ctx.moveTo(padding.left, gradYAxisZero);
    ctx.lineTo(padding.left + plotWidth, gradYAxisZero);
    ctx.stroke();

    // Y-axis grid lines and labels for gradient plot
    var gradTicks = [-1, -0.5, 0, 0.5, 1];
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    gradTicks.forEach(function (grad) {
      var y = secondPlotTop + plotHeight / 2 - grad * gradYScale;

      // Grid line
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + plotWidth, y);
      ctx.stroke();

      // Label
      ctx.fillText(grad.toFixed(1), padding.left - 10, y);
    });

    // Draw gradient curves
    // Curve for r=0: gradient = sigmoid(z)
    ctx.strokeStyle = '#0066cc';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var _x2 = 0; _x2 <= plotWidth; _x2 += 1) {
      var _z2 = zMin + _x2 / plotWidth * zRange;
      var grad = sigmoid(_z2);
      var _plotX2 = padding.left + _x2;
      var _plotY2 = secondPlotTop + plotHeight / 2 - grad * gradYScale;
      if (_x2 === 0) {
        ctx.moveTo(_plotX2, _plotY2);
      } else {
        ctx.lineTo(_plotX2, _plotY2);
      }
    }
    ctx.stroke();

    // Curve for r=1: gradient = sigmoid(z) - 1
    ctx.strokeStyle = '#cc3300';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var _x3 = 0; _x3 <= plotWidth; _x3 += 1) {
      var _z3 = zMin + _x3 / plotWidth * zRange;
      var _grad = sigmoid(_z3) - 1;
      var _plotX3 = padding.left + _x3;
      var _plotY3 = secondPlotTop + plotHeight / 2 - _grad * gradYScale;
      if (_x3 === 0) {
        ctx.moveTo(_plotX3, _plotY3);
      } else {
        ctx.lineTo(_plotX3, _plotY3);
      }
    }
    ctx.stroke();

    // Add axes labels
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // X-axis label
    ctx.fillText('Logit (z)', padding.left + plotWidth / 2, secondPlotTop + plotHeight + 35);

    // Y-axis label for loss plot
    ctx.save();
    ctx.translate(padding.left - 40, padding.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Loss', 0, 0);
    ctx.restore();

    // Y-axis label for gradient plot
    ctx.save();
    ctx.translate(padding.left - 40, secondPlotTop + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Gradient (∂L/∂z)', 0, 0);
    ctx.restore();

    // Plot titles
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Cross-Entropy Loss', padding.left + plotWidth / 2, padding.top - 20);
    ctx.fillText('Gradient of Loss', padding.left + plotWidth / 2, secondPlotTop - 20);

    // Add legend with more space
    var legendX = padding.left + plotWidth + 25;
    var legendY = padding.top + 20;
    var legendSpacing = 25;

    // Legend item for r=0
    ctx.strokeStyle = '#0066cc';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(legendX, legendY);
    ctx.lineTo(legendX + 30, legendY);
    ctx.stroke();
    ctx.fillStyle = '#0066cc';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.font = '14px Arial';
    ctx.fillText('r = 0', legendX + 40, legendY);

    // Legend item for r=1
    ctx.strokeStyle = '#cc3300';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(legendX, legendY + legendSpacing);
    ctx.lineTo(legendX + 30, legendY + legendSpacing);
    ctx.stroke();
    ctx.fillStyle = '#cc3300';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.font = '14px Arial';
    ctx.fillText('r = 1', legendX + 40, legendY + legendSpacing);

    // Add explanatory notes with better spacing
    ctx.fillStyle = '#555';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    var notes = ['For r=0:', '  Loss = -log(1-σ(z))', '  Gradient = σ(z)', '', 'For r=1:', '  Loss = -log(σ(z))', '  Gradient = σ(z)-1', '', 'where σ(z) = 1/(1+e^-z)', '', 'Note: Loss increases to ∞ as:', '• z → +∞ when r=0', '• z → -∞ when r=1'];
    notes.forEach(function (note, index) {
      ctx.fillText(note, legendX, legendY + 3 * legendSpacing + index * 20);
    });
  }, []);
  return /*#__PURE__*/React.createElement("div", {
    className: "w-full flex flex-col items-center"
  }, /*#__PURE__*/React.createElement("canvas", {
    ref: canvasRef,
    width: 840,
    height: 650,
    className: "border border-gray-300 bg-white"
  }));
};

/* export removed */ /* CrossEntropyLossAndGradient will be assigned to window.CrossEntropyLossAndGradient */
// Global export
if (typeof window !== 'undefined') {
  window.CrossEntropyVisualizer = CrossEntropyVisualizer;
}


    // Make sure the component is exported to the global window object
    if (typeof window !== 'undefined' && typeof CrossEntropyVisualizer !== 'undefined') {
        window.CrossEntropyVisualizer = CrossEntropyVisualizer;
        console.log('Successfully exported CrossEntropyVisualizer to global scope');
    }
})();
