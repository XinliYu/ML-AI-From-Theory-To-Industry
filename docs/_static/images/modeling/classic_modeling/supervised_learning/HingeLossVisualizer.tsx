import React, { useRef, useEffect, useState } from 'react';

const HingeLossVisualizer = () => {
  const canvasRef = useRef(null);
  const [margin, setMargin] = useState(1.0);

  // Hinge loss function
  const hingeLoss = (y, yHat, margin) => {
    return Math.max(0, margin - y * yHat);
  };

  // Gradient of hinge loss with respect to yHat
  const hingeGradient = (y, yHat, margin) => {
    // Gradient is -y when within margin, 0 otherwise
    return (y * yHat < margin) ? -y : 0;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Define padding and plot dimensions
    const padding = { left: 60, right: 220, top: 80, bottom: 50 };
    const plotHeight = (height - padding.top - padding.bottom - 60) / 2; // Space between plots
    const plotWidth = width - padding.left - padding.right;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);

    // Define the range for the model output (yHat)
    const yHatMin = -3;
    const yHatMax = 3;
    const yHatRange = yHatMax - yHatMin;

    // Define the range for the loss
    const lossMin = 0;
    const lossMax = 4;
    const lossRange = lossMax - lossMin;

    // Define the range for the gradient
    const gradMin = -1.1;
    const gradMax = 1.1;
    const gradRange = gradMax - gradMin;

    // Scales for mapping data to pixels
    const xScale = plotWidth / yHatRange;
    const lossYScale = plotHeight / lossRange;
    const gradYScale = plotHeight / gradRange;

    // Location of second plot (gradient)
    const secondPlotTop = padding.top + plotHeight + 60; // Space between plots

    // Draw title
    ctx.font = 'bold 18px Arial';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#333';
    ctx.fillText(`Hinge Loss and Gradient (margin = ${margin})`, width / 2, 30);

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
    const xTicks = [-3, -2, -1, 0, 1, 2, 3];

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.font = '12px Arial';
    ctx.fillStyle = '#333';

    xTicks.forEach(yHat => {
      const x = padding.left + (yHat - yHatMin) * xScale;

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
    const lossTicks = [0, 1, 2, 3, 4];

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

    // Draw vertical lines at the margin positions
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    // Margin for y = 1 (at x = margin)
    const marginPos = padding.left + (margin - yHatMin) * xScale;
    ctx.beginPath();
    ctx.moveTo(marginPos, padding.top);
    ctx.lineTo(marginPos, padding.top + plotHeight);
    ctx.stroke();

    // Margin for y = -1 (at x = -margin)
    const negMarginPos = padding.left + (-margin - yHatMin) * xScale;
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

    for (let x = 0; x <= plotWidth; x += 1) {
      const yHat = yHatMin + (x / plotWidth) * yHatRange;
      const loss = hingeLoss(1, yHat, margin);

      const plotX = padding.left + x;
      const plotY = padding.top + plotHeight - (loss - lossMin) * lossYScale;

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

    for (let x = 0; x <= plotWidth; x += 1) {
      const yHat = yHatMin + (x / plotWidth) * yHatRange;
      const loss = hingeLoss(-1, yHat, margin);

      const plotX = padding.left + x;
      const plotY = padding.top + plotHeight - (loss - lossMin) * lossYScale;

      if (x === 0) {
        ctx.moveTo(plotX, plotY);
      } else {
        ctx.lineTo(plotX, plotY);
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

    for (let x = 0; x <= plotWidth; x += 1) {
      const yHat = yHatMin + (x / plotWidth) * yHatRange;
      const grad = hingeGradient(1, yHat, margin);

      const plotX = padding.left + x;
      const plotY = secondPlotTop + plotHeight / 2 - (grad * gradYScale);

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

    for (let x = 0; x <= plotWidth; x += 1) {
      const yHat = yHatMin + (x / plotWidth) * yHatRange;
      const grad = hingeGradient(-1, yHat, margin);

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
    const legendX = padding.left + plotWidth + 20;
    const legendY = padding.top + 20;
    const legendSpacing = 25;

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

    const notes = [
      'Hinge Loss:',
      `L(ŷ, y) = max(0, ${margin} - y·ŷ)`,
      '',
      'Key properties:',
      '• Zero loss when y·ŷ ≥ margin',
      '• Linear penalty when y·ŷ < margin',
      '• Label y ∈ {-1, +1} (not 0/1)',
      '',
      'Gradient:',
      '• -y when y·ŷ < margin',
      '• 0 when y·ŷ ≥ margin (zero-gradient region)',
      '',
      'Zero-gradient region prevents overfitting',
      'as well-classified examples stop influencing',
      'the model once they exceed the margin.'
    ];

    notes.forEach((note, index) => {
      ctx.fillText(note, legendX, legendY + 3 * legendSpacing + 10 + index * 20);
    });

  }, [margin]);

  return (
    <div className="w-full flex flex-col items-center">
      <div className="mb-4 flex items-center justify-center gap-4">
        <label htmlFor="margin" className="font-medium">Margin:</label>
        <input
          id="margin"
          type="range"
          min="0.2"
          max="2.0"
          step="0.1"
          value={margin}
          onChange={(e) => setMargin(parseFloat(e.target.value))}
          className="w-40"
        />
        <span className="w-10 text-center">{margin.toFixed(1)}</span>
      </div>
      <canvas
        ref={canvasRef}
        width={850}
        height={650}
        className="border border-gray-300 bg-white"
      />
    </div>
  );
};

export default HingeLossVisualizer;