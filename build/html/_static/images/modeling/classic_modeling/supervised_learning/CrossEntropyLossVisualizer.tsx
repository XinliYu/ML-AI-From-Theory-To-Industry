import React, { useRef, useEffect } from 'react';

const CrossEntropyLossVisualizer = () => {
  const canvasRef = useRef(null);

  // Sigmoid function
  const sigmoid = (z) => {
    return 1 / (1 + Math.exp(-z));
  };

  // Binary cross-entropy loss function
  const binaryCrossEntropy = (z, r) => {
    const p = sigmoid(z);
    // Avoid numerical instability with a safer implementation
    if (r === 1) {
      // For r=1, use -log(p) with a more numerically stable approach for large negative z
      return z >= 0 ? Math.log(1 + Math.exp(-z)) : -z + Math.log(1 + Math.exp(z));
    } else {
      // For r=0, use -log(1-p) with a more numerically stable approach for large positive z
      return z >= 0 ? z + Math.log(1 + Math.exp(-z)) : Math.log(1 + Math.exp(z));
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Define padding and plot dimensions with more space between plots
    const padding = { left: 60, right: 200, top: 80, bottom: 50 };
    const plotHeight = (height - padding.top - padding.bottom - 60) / 2; // More space between plots
    const plotWidth = width - padding.left - padding.right;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);

    // Define the range for the logit z
    const zMin = -6;
    const zMax = 6;
    const zRange = zMax - zMin;

    // Define the range for the loss (higher range to show more of the curve)
    const lossMin = 0;
    const lossMax = 8;
    const lossRange = lossMax - lossMin;

    // Define the range for the gradient
    const gradMin = -1.1;
    const gradMax = 1.1;
    const gradRange = gradMax - gradMin;

    // Scales for mapping data to pixels
    const xScale = plotWidth / zRange;
    const lossYScale = plotHeight / lossRange;
    const gradYScale = plotHeight / gradRange;

    // Location of second plot (gradient) with more space
    const secondPlotTop = padding.top + plotHeight + 60; // Increased space between plots

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
    const xTicks = [-6, -4, -2, 0, 2, 4, 6];

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.font = '12px Arial';
    ctx.fillStyle = '#333';

    xTicks.forEach(z => {
      const x = padding.left + (z - zMin) * xScale;

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
    const lossTicks = [0, 2, 4, 6, 8];

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    lossTicks.forEach(loss => {
      const y = padding.top + plotHeight - (loss - lossMin) * lossYScale;

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

    let lastValidY0 = null;

    for (let x = 0; x <= plotWidth; x += 1) {
      const z = zMin + (x / plotWidth) * zRange;
      let loss = binaryCrossEntropy(z, 0);
      if (isNaN(loss) || !isFinite(loss)) {
        loss = lossMax; // Cap at max value
      }

      const plotX = padding.left + x;
      const rawY = padding.top + plotHeight - (loss - lossMin) * lossYScale;
      const plotY = Math.max(padding.top, rawY); // Ensure we stay within plot bounds

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

    let lastValidY1 = null;

    for (let x = 0; x <= plotWidth; x += 1) {
      const z = zMin + (x / plotWidth) * zRange;
      let loss = binaryCrossEntropy(z, 1);
      if (isNaN(loss) || !isFinite(loss)) {
        loss = lossMax; // Cap at max value
      }

      const plotX = padding.left + x;
      const rawY = padding.top + plotHeight - (loss - lossMin) * lossYScale;
      const plotY = Math.max(padding.top, rawY); // Ensure we stay within plot bounds

      if (x === 0 || lastValidY1 === null) {
        ctx.moveTo(plotX, plotY);
        lastValidY1 = plotY;
      } else {
        // If the value jumps too much, draw a vertical line to indicate discontinuity
        if (Math.abs(plotY - lastValidY1) > 50 && plotY === padding.top) {
          ctx.lineTo(plotX, lastValidY1);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(plotX, plotY);
        } else {
          ctx.lineTo(plotX, plotY);
        }
        lastValidY1 = plotY;
      }
    }

    ctx.stroke();

    // Add a visual indication that the curve continues upward
    // For r=0 at right side
    const rightEdgeX = padding.left + plotWidth;
    const rightEdgeY0 = padding.top + plotHeight - (binaryCrossEntropy(zMax, 0) - lossMin) * lossYScale;
    if (rightEdgeY0 <= padding.top + 20) {
        ctx.strokeStyle = '#0066cc';
        ctx.beginPath();
        ctx.moveTo(rightEdgeX - 10, padding.top + 10);
        ctx.lineTo(rightEdgeX, padding.top);
        ctx.lineTo(rightEdgeX - 10, padding.top - 10);
        ctx.stroke();
    }

    // For r=1 at left side
    const leftEdgeX = padding.left;
    const leftEdgeY1 = padding.top + plotHeight - (binaryCrossEntropy(zMin, 1) - lossMin) * lossYScale;
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
    const gradYAxisZero = secondPlotTop + plotHeight / 2;
    ctx.moveTo(padding.left, gradYAxisZero);
    ctx.lineTo(padding.left + plotWidth, gradYAxisZero);

    ctx.stroke();

    // Y-axis grid lines and labels for gradient plot
    const gradTicks = [-1, -0.5, 0, 0.5, 1];

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    gradTicks.forEach(grad => {
      const y = secondPlotTop + plotHeight / 2 - (grad * gradYScale);

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

    for (let x = 0; x <= plotWidth; x += 1) {
      const z = zMin + (x / plotWidth) * zRange;
      const grad = sigmoid(z);
      const plotX = padding.left + x;
      const plotY = secondPlotTop + plotHeight / 2 - (grad * gradYScale);

      if (x === 0) {
        ctx.moveTo(plotX, plotY);
      } else {
        ctx.lineTo(plotX, plotY);
      }
    }

    ctx.stroke();

    // Curve for r=1: gradient = sigmoid(z) - 1
    ctx.strokeStyle = '#cc3300';
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let x = 0; x <= plotWidth; x += 1) {
      const z = zMin + (x / plotWidth) * zRange;
      const grad = sigmoid(z) - 1;
      const plotX = padding.left + x;
      const plotY = secondPlotTop + plotHeight / 2 - (grad * gradYScale);

      if (x === 0) {
        ctx.moveTo(plotX, plotY);
      } else {
        ctx.lineTo(plotX, plotY);
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
    const legendX = padding.left + plotWidth + 25;
    const legendY = padding.top + 20;
    const legendSpacing = 25;

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

    const notes = [
      'For r=0:',
      '  Loss = -log(1-σ(z))',
      '  Gradient = σ(z)',
      '',
      'For r=1:',
      '  Loss = -log(σ(z))',
      '  Gradient = σ(z)-1',
      '',
      'where σ(z) = 1/(1+e^-z)',
      '',
      'Note: Loss increases to ∞ as:',
      '• z → +∞ when r=0',
      '• z → -∞ when r=1'
    ];

    notes.forEach((note, index) => {
      ctx.fillText(note, legendX, legendY + 3 * legendSpacing + index * 20);
    });

  }, []);

  return (
    <div className="w-full flex flex-col items-center">
      <canvas
        ref={canvasRef}
        width={840}
        height={650}
        className="border border-gray-300 bg-white"
      />
    </div>
  );
};

export default CrossEntropyLossVisualizer;