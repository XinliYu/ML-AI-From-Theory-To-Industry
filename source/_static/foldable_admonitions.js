document.addEventListener("DOMContentLoaded", function () {
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
});
