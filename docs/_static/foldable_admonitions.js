document.addEventListener("DOMContentLoaded", function () {
    // Handle collapsible admonitions
    document.querySelectorAll(".admonition").forEach(function (admonition) {
        let title = admonition.querySelector(".admonition-title");
        if (title) {
            title.style.cursor = "pointer";
            // Select everything inside the admonition *except* the title
            let contentElements = Array.from(admonition.children).filter(el => !el.classList.contains("admonition-title"));
            // Default: Expanded (ensure visible)
            admonition.classList.remove("collapsed");
            contentElements.forEach(el => el.style.display = "block");
            // Toggle collapsibility on click
            title.addEventListener("click", function () {
                let isCollapsed = admonition.classList.toggle("collapsed");
                contentElements.forEach(el => el.style.display = isCollapsed ? "none" : "block");
            });
        }
    });

    // Handle collapsible code blocks
    document.querySelectorAll(".folding").forEach(function(block) {
        // Get the name attribute if set
        const name = block.getAttribute("title") || block.getAttribute("name") || block.getAttribute("id") || "Code";
        
        // Create wrapper elements
        const wrapper = document.createElement("div");
        wrapper.className = "code-folding-wrapper";
        
        const header = document.createElement("div");
        header.className = "code-folding-header";
        header.innerHTML = '<span class="code-folding-toggle">+</span> <strong>' + name + '</strong>';
        
        const content = document.createElement("div");
        content.className = "code-folding-content";
        content.style.display = "none";
        
        // Move the code block into the content div
        block.parentNode.insertBefore(wrapper, block);
        content.appendChild(block);
        wrapper.appendChild(header);
        wrapper.appendChild(content);
        
        // Add click event
        header.addEventListener("click", function() {
            const isVisible = content.style.display !== "none";
            content.style.display = isVisible ? "none" : "block";
            header.querySelector(".code-folding-toggle").textContent = isVisible ? "+" : "-";
        });
    });
});