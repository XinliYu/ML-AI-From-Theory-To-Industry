
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
function _slicedToArray(r, e) { return _arrayWithHoles(r) || _iterableToArrayLimit(r, e) || _unsupportedIterableToArray(r, e) || _nonIterableRest(); }
function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }
function _unsupportedIterableToArray(r, a) { if (r) { if ("string" == typeof r) return _arrayLikeToArray(r, a); var t = {}.toString.call(r).slice(8, -1); return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray(r, a) : void 0; } }
function _arrayLikeToArray(r, a) { (null == a || a > r.length) && (a = r.length); for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e]; return n; }
function _iterableToArrayLimit(r, l) { var t = null == r ? null : "undefined" != typeof Symbol && r[Symbol.iterator] || r["@@iterator"]; if (null != t) { var e, n, i, u, a = [], f = !0, o = !1; try { if (i = (t = t.call(r)).next, 0 === l) { if (Object(t) !== t) return; f = !1; } else for (; !(f = (e = i.call(t)).done) && (a.push(e.value), a.length !== l); f = !0); } catch (r) { o = !0, n = r; } finally { try { if (!f && null != t.return && (u = t.return(), Object(u) !== u)) return; } finally { if (o) throw n; } } return a; } }
function _arrayWithHoles(r) { if (Array.isArray(r)) return r; }
// Using global React object (window.React) instead of imports

var HingeLossVisualizer = function HingeLossVisualizer() {
  var canvasRef = useRef(null);
  var _useState = useState(1.0),
    _useState2 = _slicedToArray(_useState, 2),
    margin = _useState2[0],
    setMargin = _useState2[1];

  // Hinge loss function
  var hingeLoss = function hingeLoss(y, yHat, margin) {
    return Math.max(0, margin - y * yHat);
  };

  // Gradient of hinge loss with respect to yHat
  var hingeGradient = function hingeGradient(y, yHat, margin) {
    // Gradient is -y when within margin, 0 otherwise
    return y * yHat < margin ? -y : 0;
  };
  useEffect(function () {
    var canvas = canvasRef.current;
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var width = canvas.width;
    var height = canvas.height;

    // Define padding and plot dimensions
    var padding = {
      left: 60,
      right: 220,
      top: 80,
      bottom: 50
    };
    var plotHeight = (height - padding.top - padding.bottom - 60) / 2; // Space between plots
    var plotWidth = width - padding.left - padding.right;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);

    // Define the range for the model output (yHat)
    var yHatMin = -3;
    var yHatMax = 3;
    var yHatRange = yHatMax - yHatMin;

    // Define the range for the loss
    var lossMin = 0;
    var lossMax = 4;
    var lossRange = lossMax - lossMin;

    // Define the range for the gradient
    var gradMin = -1.1;
    var gradMax = 1.1;
    var gradRange = gradMax - gradMin;

    // Scales for mapping data to pixels
    var xScale = plotWidth / yHatRange;
    var lossYScale = plotHeight / lossRange;
    var gradYScale = plotHeight / gradRange;

    // Location of second plot (gradient)
    var secondPlotTop = padding.top + plotHeight + 60; // Space between plots

    // Draw title
    ctx.font = 'bold 18px Arial';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#333';
    ctx.fillText("Hinge Loss and Gradient (margin = ".concat(margin, ")"), width / 2, 30);

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
    var xTicks = [-3, -2, -1, 0, 1, 2, 3];
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.font = '12px Arial';
    ctx.fillStyle = '#333';
    xTicks.forEach(function (yHat) {
      var x = padding.left + (yHat - yHatMin) * xScale;

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
      ctx.fillText(yHat.toString(), x, secondPlotTop + plotHeight + 10);
    });

    // Y-axis grid lines and labels for loss plot
    var lossTicks = [0, 1, 2, 3, 4];
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

    // Draw vertical lines at the margin positions
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    // Margin for y = 1 (at x = margin)
    var marginPos = padding.left + (margin - yHatMin) * xScale;
    ctx.beginPath();
    ctx.moveTo(marginPos, padding.top);
    ctx.lineTo(marginPos, padding.top + plotHeight);
    ctx.stroke();

    // Margin for y = -1 (at x = -margin)
    var negMarginPos = padding.left + (-margin - yHatMin) * xScale;
    ctx.beginPath();
    ctx.moveTo(negMarginPos, padding.top);
    ctx.lineTo(negMarginPos, padding.top + plotHeight);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw loss curves
    // Curve for y = 1
    ctx.strokeStyle = '#0066cc';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var x = 0; x <= plotWidth; x += 1) {
      var yHat = yHatMin + x / plotWidth * yHatRange;
      var loss = hingeLoss(1, yHat, margin);
      var plotX = padding.left + x;
      var plotY = padding.top + plotHeight - (loss - lossMin) * lossYScale;
      if (x === 0) {
        ctx.moveTo(plotX, plotY);
      } else {
        ctx.lineTo(plotX, plotY);
      }
    }
    ctx.stroke();

    // Curve for y = -1
    ctx.strokeStyle = '#cc3300';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var _x = 0; _x <= plotWidth; _x += 1) {
      var _yHat = yHatMin + _x / plotWidth * yHatRange;
      var _loss = hingeLoss(-1, _yHat, margin);
      var _plotX = padding.left + _x;
      var _plotY = padding.top + plotHeight - (_loss - lossMin) * lossYScale;
      if (_x === 0) {
        ctx.moveTo(_plotX, _plotY);
      } else {
        ctx.lineTo(_plotX, _plotY);
      }
    }
    ctx.stroke();

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

    // Draw vertical lines at the margin positions (for gradient plot)
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    // Margin for y = 1 (at x = margin)
    ctx.beginPath();
    ctx.moveTo(marginPos, secondPlotTop);
    ctx.lineTo(marginPos, secondPlotTop + plotHeight);
    ctx.stroke();

    // Margin for y = -1 (at x = -margin)
    ctx.beginPath();
    ctx.moveTo(negMarginPos, secondPlotTop);
    ctx.lineTo(negMarginPos, secondPlotTop + plotHeight);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw gradient curves
    // Curve for y = 1
    ctx.strokeStyle = '#0066cc';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var _x2 = 0; _x2 <= plotWidth; _x2 += 1) {
      var _yHat2 = yHatMin + _x2 / plotWidth * yHatRange;
      var grad = hingeGradient(1, _yHat2, margin);
      var _plotX2 = padding.left + _x2;
      var _plotY2 = secondPlotTop + plotHeight / 2 - grad * gradYScale;
      if (_x2 === 0) {
        ctx.moveTo(_plotX2, _plotY2);
      } else {
        ctx.lineTo(_plotX2, _plotY2);
      }
    }
    ctx.stroke();

    // Curve for y = -1
    ctx.strokeStyle = '#cc3300';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var _x3 = 0; _x3 <= plotWidth; _x3 += 1) {
      var _yHat3 = yHatMin + _x3 / plotWidth * yHatRange;
      var _grad = hingeGradient(-1, _yHat3, margin);
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
    ctx.fillText('Model Output (ŷ)', padding.left + plotWidth / 2, secondPlotTop + plotHeight + 35);

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
    ctx.fillText('Gradient (∂L/∂ŷ)', 0, 0);
    ctx.restore();

    // Plot titles
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Hinge Loss', padding.left + plotWidth / 2, padding.top - 20);
    ctx.fillText('Gradient of Loss', padding.left + plotWidth / 2, secondPlotTop - 20);

    // Add legend
    var legendX = padding.left + plotWidth + 20;
    var legendY = padding.top + 20;
    var legendSpacing = 25;

    // Legend item for y = 1
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
    ctx.fillText('y = +1 (positive class)', legendX + 40, legendY);

    // Legend item for y = -1
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
    ctx.fillText('y = -1 (negative class)', legendX + 40, legendY + legendSpacing);

    // Add dashed line to legend for margin
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(legendX, legendY + 2 * legendSpacing);
    ctx.lineTo(legendX + 30, legendY + 2 * legendSpacing);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#888';
    ctx.fillText('margin boundary', legendX + 40, legendY + 2 * legendSpacing);

    // Add explanatory notes
    ctx.fillStyle = '#555';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    var notes = ['Hinge Loss:', "L(\u0177, y) = max(0, ".concat(margin, " - y\xB7\u0177)"), '', 'Key properties:', '• Zero loss when y·ŷ ≥ margin', '• Linear penalty when y·ŷ < margin', '• Label y ∈ {-1, +1} (not 0/1)', '', 'Gradient:', '• -y when y·ŷ < margin', '• 0 when y·ŷ ≥ margin (zero-gradient region)', '', 'Zero-gradient region prevents overfitting', 'as well-classified examples stop influencing', 'the model once they exceed the margin.'];
    notes.forEach(function (note, index) {
      ctx.fillText(note, legendX, legendY + 3 * legendSpacing + 10 + index * 20);
    });
  }, [margin]);
  return /*#__PURE__*/React.createElement("div", {
    className: "w-full flex flex-col items-center"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mb-4 flex items-center justify-center gap-4"
  }, /*#__PURE__*/React.createElement("label", {
    htmlFor: "margin",
    className: "font-medium"
  }, "Margin:"), /*#__PURE__*/React.createElement("input", {
    id: "margin",
    type: "range",
    min: "0.2",
    max: "2.0",
    step: "0.1",
    value: margin,
    onChange: function onChange(e) {
      return setMargin(parseFloat(e.target.value));
    },
    className: "w-40"
  }), /*#__PURE__*/React.createElement("span", {
    className: "w-10 text-center"
  }, margin.toFixed(1))), /*#__PURE__*/React.createElement("canvas", {
    ref: canvasRef,
    width: 850,
    height: 650,
    className: "border border-gray-300 bg-white"
  }));
};

/* export removed */ /* HingeLossVisualizer will be assigned to window.HingeLossVisualizer */
// Global export
if (typeof window !== 'undefined') {
  window.HingeLossVisualizer = HingeLossVisualizer;
}


    // Make sure the component is exported to the global window object
    if (typeof window !== 'undefined' && typeof HingeLossVisualizer !== 'undefined') {
        window.HingeLossVisualizer = HingeLossVisualizer;
        console.log('Successfully exported HingeLossVisualizer to global scope');
    }
})();
