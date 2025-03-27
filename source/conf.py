from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxRole, SphinxDirective
from docutils.parsers.rst.directives.tables import Table
from docutils.parsers.rst import Directive
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.directives.admonitions import BaseAdmonition
import re
import logging
import os
import subprocess
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MLAI'
copyright = '2025, Tony'
author = 'Tony'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.mathjax',  # For math support
    'sphinx.ext.todo',
    'sphinx.ext.viewcode'
]

# Basic MathJax config
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$']],
        'displayMath': [['$$', '$$']],
    }
}

# Enable MyST features
myst_enable_extensions = [
    "dollarmath",
    "html_image",  # For HTML images
    "colon_fence",  # For code blocks with colons
    "tasklist",  # For task lists
    "deflist",  # For definition lists
    "html_admonition",  # For admonitions,
    'sphinx.ext.autosectionlabel',  # This helps with automatic section labeling
    'sphinx.ext.intersphinx',  # This helps with cross-project references
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
myst_parse_rst = True
autosectionlabel_prefix_document = True
templates_path = ['_templates']
html_js_files = [
    'js/mathjax-config.js',
    'https://unpkg.com/react@17/umd/react.production.min.js',
    'https://unpkg.com/react-dom@17/umd/react-dom.production.min.js'
]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_use_index = True
html_baseurl = 'https://xinliyu.github.io/ML-AI-From-Theory-To-Industry/'  # Important for GitHub Pages
html_use_relative_paths = True
html_theme_options = {
    'navigation_depth': 6,  # Increase this number to show deeper levels
}
numfig = True
numfig_format = {
    'table': 'Table %s',
    'figure': 'Figure %s',
    'code-block': 'Listing %s',
    'section': 'Section %s',
}


class HtmlTable(SphinxDirective, Table):
    """
    Enhanced version of the standard table directive that supports raw HTML content.
    """

    def run(self):
        # Check if there's raw HTML content
        has_raw_html = any('.. raw:: html' in line for line in self.content)

        if not has_raw_html:
            # If no raw HTML, process as normal table
            return super().run()

        # Create a container node to properly maintain document structure
        container = nodes.container()
        container['classes'] = ['html-table-container']

        # Create a table node
        table = nodes.table()
        container += table

        # Handle title and automatic numbering
        if self.arguments:
            title_text = self.arguments[0]
            table['title'] = title_text

            # Generate the table number - use Sphinx's built-in numbering
            env = self.env
            if hasattr(env, 'docname') and self.config.numfig:
                figtype = 'table'

                # Register the table with Sphinx's figure collector
                if not hasattr(env, '_table_counter'):
                    env._table_counter = {}

                docname = env.docname
                if docname not in env._table_counter:
                    env._table_counter[docname] = 0

                table_num = env._table_counter[docname] = env._table_counter[docname] + 1

                # Format with numfig format if available
                if hasattr(self.config, 'numfig_format') and figtype in self.config.numfig_format:
                    caption_text = self.config.numfig_format[figtype] % table_num + ": " + title_text
                else:
                    caption_text = f"Table {table_num}: {title_text}"
            else:
                caption_text = title_text

            # Create caption for both the node and HTML
            caption_html = f'<div class="table-caption">{caption_text}</div>'
        else:
            caption_text = ''
            caption_html = ''

        # Handle options
        if 'name' in self.options:
            self.options['name'] = nodes.fully_normalize_name(self.options['name'])
            self.add_name(table)

        if 'class' in self.options:
            table['classes'] += self.options['class']

        if 'width' in self.options:
            table['width'] = self.options['width']

        if 'align' in self.options:
            table['align'] = self.options['align']

        # Extract the raw HTML content
        raw_content = []
        in_raw_block = False

        for i, line in enumerate(self.content):
            if '.. raw:: html' in line:
                in_raw_block = True
                continue
            elif in_raw_block and not line.strip() and i < len(self.content) - 1 and all(
                    not l.strip() for l in self.content[i:]):
                # End of content
                break

            if in_raw_block:
                # Remove the indentation (if any)
                if line.startswith('   '):  # Standard rst indentation
                    raw_content.append(line[3:])
                else:
                    raw_content.append(line)

        raw_html = '\n'.join(raw_content)

        # Create a wrapper with caption OUTSIDE the table
        if caption_html:
            # Place caption before the table HTML
            raw_html = f'''
            <div class="table-wrapper">
                {caption_html}
                {raw_html}
            </div>
            '''
        else:
            raw_html = f'''
            <div class="table-wrapper">
                {raw_html}
            </div>
            '''

        # Create raw node and add it to the table
        raw_node = nodes.raw('', raw_html, format='html')
        table += raw_node

        # Register the table in Sphinx's environment for list of tables
        if self.arguments and hasattr(env, 'docname'):
            if not hasattr(env, 'table_list'):
                env.table_list = {}
            env.table_list.setdefault((env.docname, 'table'), [])
            env.table_list[(env.docname, 'table')].append(
                (table['ids'][0] if 'ids' in table and table['ids'] else '',
                 caption_text))

        # Add a paragraph node after the table to ensure proper section separation
        after_node = nodes.paragraph()
        container += after_node

        return [container]


class CustomNoteDirective(BaseAdmonition):
    """
    Custom note directive that supports more flexible title formatting
    including math and inline roles, and prevents duplicate titles
    """

    node_class = nodes.note

    def run(self):
        # First, let the parent class create the basic structure
        admonition_node = super().run()[0]

        # Check if the admonition has a title
        if len(admonition_node) > 0 and isinstance(admonition_node[0], nodes.title):
            title_node = admonition_node[0]

            # Get the title text
            title_text = title_node.astext()

            # If it doesn't already have "Note:", add it
            if not title_text.startswith('Note:'):
                # Create a new title node
                new_title = nodes.title('', f"Note: {title_text}")

                # Replace the old title with our new one
                admonition_node.replace(title_node, new_title)

        return [admonition_node]


class NewConceptRole(SphinxRole):
    def run(self):
        text = self.text
        label = None
        bold = False

        # Check for bold option
        if text.startswith('!'):
            bold = True
            text = text[1:]  # Remove the ! prefix

        # Check if there's a manual label
        if '<' in text and text.endswith('>'):
            text, label = text.rsplit('<', 1)
            label = label[:-1]  # Remove the trailing '>'
        else:
            # Auto-generate label - remove text in parentheses, convert spaces to underscores
            label_text = text
            label_text = re.sub(r'\([^)]*\)', '', label_text)  # Remove text in parentheses
            label_text = label_text.strip()  # Remove leading/trailing whitespace
            label = label_text.replace(' ', '_').lower()  # Convert spaces to underscores and lowercase

        # Choose the appropriate node type and styling
        if bold:
            # Create a strong node with newconcept class
            node = nodes.strong(self.rawtext, '', classes=['newconcept'])
        else:
            # Create a regular inline node with newconcept class
            node = nodes.inline(self.rawtext, '', classes=['newconcept'])

        node += nodes.Text(text)

        # Create a target for linking
        target_id = f'newconcept-{label}'
        target = nodes.target('', '', ids=[target_id])

        # Register this target in the environment for cross-referencing
        env = self.env
        if not hasattr(env, 'newconcept_all_concepts'):
            env.newconcept_all_concepts = {}

        env.newconcept_all_concepts[target_id] = {
            'docname': env.docname,
            'lineno': self.lineno,
            'target_id': target_id,
            'text': text
        }

        # Make the node a reference
        self.set_source_info(target)

        return [target, node], []


def ref_newconcept_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Role for referencing a newconcept item."""
    env = inliner.document.settings.env

    # Clean the text by removing text in parentheses
    clean_text = re.sub(r'\([^)]*\)', '', text)
    clean_text = clean_text.strip()

    # Handle label transformation - convert spaces to underscores and lowercase
    label = clean_text.replace(' ', '_').lower()

    target_id = f'newconcept-{label}'

    # Create a reference node with a custom attribute to identify it later
    refnode = nodes.reference('', '')
    refnode['refid'] = target_id
    refnode['is_newconcept_ref'] = True  # Custom marker for processing

    # Add a styled inline node within the reference
    refnode += nodes.inline('', text, classes=['refconcept'])

    return [refnode], []


# Keep your other role and directive definitions...

def process_newconcept_nodes(app, doctree, fromdocname):
    """
    Process newconcept references to handle cross-document links.
    """
    env = app.builder.env

    # Skip if we have no newconcepts
    if not hasattr(env, 'newconcept_all_concepts'):
        return

    # Process all reference nodes that have our custom marker
    for node in doctree.traverse(nodes.reference):
        if node.get('is_newconcept_ref') and 'refid' in node:
            target_id = node['refid']

            if target_id in env.newconcept_all_concepts:
                concept = env.newconcept_all_concepts[target_id]
                todocname = concept['docname']

                # Update the reference URI for cross-document links
                if todocname != fromdocname:
                    node['refuri'] = app.builder.get_relative_uri(
                        fromdocname, todocname) + '#' + target_id
                    del node['refid']  # Remove refid once refuri is set
                # else keep the refid for same-document links
            else:
                # If target not found, just make it plain text
                # Use a compatible warning method
                import warnings
                warnings.warn(f'{fromdocname}: newconcept reference target not found: {target_id}')
                node.replace_self(nodes.Text(node[0].astext()))

            # Remove our custom marker
            if 'is_newconcept_ref' in node:
                del node['is_newconcept_ref']


# Properly implement environment collector
class NewConceptCollector(EnvironmentCollector):
    # Required collector attributes
    name = 'newconcept'

    def clear_doc(self, app, env, docname):
        if hasattr(env, 'newconcept_all_concepts'):
            env.newconcept_all_concepts = {
                target_id: info for target_id, info in env.newconcept_all_concepts.items()
                if info['docname'] != docname
            }

    def merge_other(self, app, env, docnames, other):
        if hasattr(other, 'newconcept_all_concepts'):
            if not hasattr(env, 'newconcept_all_concepts'):
                env.newconcept_all_concepts = {}
            for target_id, info in other.newconcept_all_concepts.items():
                if info['docname'] in docnames:
                    env.newconcept_all_concepts[target_id] = info

    def process_doc(self, app, doctree):
        pass  # Processing happens in the NewConceptRole


def underline_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    node = nodes.inline(rawtext, text, classes=['underline'])
    return [node], []


def underline_bold_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    """Role for both underlined and bold text."""
    # Create a strong node (bold) with the underline-bold class
    node = nodes.strong(rawtext, text, classes=['underline-bold'])
    return [node], []


# Custom admonition for examples with green styling
class CustomAdmonition(SphinxDirective):
    """Custom admonition directive with colored background."""

    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        'class': directives.class_option,
        'name': directives.unchanged,
    }

    def run(self):
        # Use the first argument as the title
        title_text = self.arguments[0]

        # Set the default class to 'admonition'
        node_class = ['admonition']
        if self.options.get('class'):
            node_class.extend(self.options.get('class'))
        else:
            # Add example-green class by default
            node_class.append('example-green')

        # Create the admonition node
        admonition_node = nodes.admonition('', classes=node_class)

        # Create the title node
        title = nodes.title(title_text, title_text)
        admonition_node += title

        # Parse the content and add it to the admonition
        self.state.nested_parse(self.content, self.content_offset, admonition_node)

        return [admonition_node]


def convert_tsx_to_js(app):
    """
    Find all .tsx files in _static/images directory, convert them to JS using Babel
    and make them compatible with browser environments.
    Preserves the full directory structure and properly handles exports.
    """
    # Path to directories
    static_images_dir = os.path.join(app.confdir, '_static', 'images')
    static_js_dir = os.path.join(app.confdir, '_static', 'js')

    # Ensure the JS directory exists
    os.makedirs(static_js_dir, exist_ok=True)

    # Find all .tsx files recursively
    tsx_files = []
    for root, _, files in os.walk(static_images_dir):
        for file in files:
            if file.endswith('.tsx'):
                tsx_files.append(os.path.join(root, file))

    if not tsx_files:
        print("No TSX files found to convert.")
        return

    print(f"Found {len(tsx_files)} TSX files to convert.")

    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple package.json for npm
        package_json_path = os.path.join(temp_dir, 'package.json')
        with open(package_json_path, 'w') as f:
            f.write('''{
                "name": "tsx-converter",
                "version": "1.0.0",
                "private": true
            }''')

        # Create babel config for classic JSX transform (not automatic)
        babel_config_path = os.path.join(temp_dir, 'babel.config.json')
        with open(babel_config_path, 'w') as f:
            f.write('''{
                "presets": [
                    ["@babel/preset-env", { 
                        "targets": { "browsers": "last 2 versions" },
                        "modules": false
                    }],
                    ["@babel/preset-react", { 
                        "runtime": "classic",
                        "pragma": "React.createElement"
                    }],
                    "@babel/preset-typescript"
                ]
            }''')

        # Install Babel dependencies
        try:
            print("Installing Babel dependencies...")
            subprocess.run([
                'npm', 'install', '--no-save',
                '@babel/core', '@babel/cli', '@babel/preset-env',
                '@babel/preset-react', '@babel/preset-typescript'
            ], check=True, cwd=temp_dir)
        except Exception as e:
            print(f"Error installing Babel dependencies: {e}")
            return

        # Process each TSX file
        for tsx_file in tsx_files:
            # Keep the full relative path structure
            rel_path = os.path.relpath(tsx_file, static_images_dir)

            # Create temp file path with the same structure
            temp_tsx_path = os.path.join(temp_dir, rel_path)

            # Create js output path with the same structure
            js_rel_path = rel_path.replace('.tsx', '.js')
            js_out_path = os.path.join(static_js_dir, js_rel_path)

            # Create directory structure if needed
            os.makedirs(os.path.dirname(temp_tsx_path), exist_ok=True)
            os.makedirs(os.path.dirname(js_out_path), exist_ok=True)

            # Read the original TSX
            with open(tsx_file, 'r') as f:
                tsx_content = f.read()

            # Convert any import statements to use window.React instead
            modified_tsx = re.sub(
                r'import React,\s*{\s*([^}]+)\s*}\s*from\s*[\'"]react[\'"];',
                r'// Using global React object (window.React) instead of imports',
                tsx_content
            )

            # Remove export statements or convert them to global window assignments
            modified_tsx = re.sub(
                r'export\s+default\s+(\w+)\s*;?',
                r'/* export removed */ /* \1 will be assigned to window.\1 */',
                modified_tsx
            )

            # Get component name from filename (preserve case)
            component_name = os.path.basename(tsx_file).replace('.tsx', '')

            # Add the export code at the end
            export_code = f"""
// Global export
if (typeof window !== 'undefined') {{
    window.{component_name} = {component_name};
}}
"""
            modified_tsx = modified_tsx + export_code

            # Write the modified TSX
            with open(temp_tsx_path, 'w') as f:
                f.write(modified_tsx)

            # Use Babel to convert TSX to JS directly
            try:
                print(f"Converting {rel_path} using Babel...")
                babel_bin = os.path.join(temp_dir, 'node_modules', '.bin', 'babel')
                babel_output = subprocess.run(
                    [babel_bin, temp_tsx_path, '--out-file', '/dev/stdout'],
                    check=True, cwd=temp_dir, capture_output=True, text=True
                ).stdout

                # Remove any remaining export statements that Babel might have kept
                babel_output = re.sub(
                    r'(?:export\s+(?:default|var|let|const)\s+[^;]+;|exports\.default\s*=\s*[^;]+;)',
                    '/* exports removed */',
                    babel_output
                )

                # Create browser-compatible wrapper that provides React variables
                js_content = f"""
// Auto-converted from TSX using Babel - Browser Compatible Version
(function() {{
    // Get React from global scope
    var React = window.React;

    // Make React hooks available
    var useState = React.useState;
    var useEffect = React.useEffect;
    var useRef = React.useRef;

    // Simple JSX runtime replacement
    var jsx = function(type, props, key, children) {{
        var newProps = props || {{}};
        if (children !== undefined) {{
            newProps.children = children;
        }}
        return React.createElement(type, newProps, children);
    }};

    var jsxs = function(type, props, key, children) {{
        return jsx(type, props, key, children);
    }};

    // Helper functions from Babel
    function _extends() {{
        _extends = Object.assign || function(target) {{
            for (var i = 1; i < arguments.length; i++) {{
                var source = arguments[i];
                for (var key in source) {{
                    if (Object.prototype.hasOwnProperty.call(source, key)) {{
                        target[key] = source[key];
                    }}
                }}
            }}
            return target;
        }};
        return _extends.apply(this, arguments);
    }}

    // Babel transpiled component with export statements removed
{babel_output.replace('require("react")', '/* React */')
                .replace('require("react/jsx-runtime")', '/* JSX Runtime */')
                .replace('_jsxRuntime.jsx', 'jsx')
                .replace('_jsxRuntime.jsxs', 'jsxs')
                .replace('_react.default', 'React')
                .replace('_react.useRef', 'useRef')
                .replace('_react.useState', 'useState')
                .replace('_react.useEffect', 'useEffect')}

    // Make sure the component is exported to the global window object
    if (typeof window !== 'undefined' && typeof {component_name} !== 'undefined') {{
        window.{component_name} = {component_name};
        console.log('Successfully exported {component_name} to global scope');
    }}
}})();
"""

                # Write the processed JS
                with open(js_out_path, 'w') as f:
                    f.write(js_content)

                print(f"Successfully converted: {js_rel_path}")

            except subprocess.CalledProcessError as e:
                print(f"Error converting {rel_path}: {e}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")

        print("TSX to JS conversion completed.")


class ReactComponentDirective(Directive):
    required_arguments = 1  # Path to the TSX file
    optional_arguments = 0
    has_content = True  # Allow content for props
    option_spec = {
        'width': directives.unchanged,
        'height': directives.unchanged,
        'align': directives.unchanged,
        'max-width': directives.unchanged,
        'min-width': directives.unchanged,
        'class': directives.unchanged,
        'responsive': directives.flag,
        'center': directives.flag,
        'katex': directives.flag  # Option to enable KaTeX
    }

    def run(self):
        env = self.state.document.settings.env
        tsx_path = self.arguments[0]

        # Get styling options from directive or use defaults
        width = self.options.get('width', 'auto')
        height = self.options.get('height', 'auto')
        align = self.options.get('align', 'center')
        max_width = self.options.get('max-width', '100%')
        min_width = self.options.get('min-width', 'auto')
        custom_class = self.options.get('class', '')
        is_responsive = 'responsive' in self.options
        is_centered = 'center' in self.options or align == 'center'
        use_katex = 'katex' in self.options

        # Convert path from _static/images/... to _static/js/... PRESERVING SUBDIRECTORIES
        if '_static/images/' in tsx_path:
            # This is crucial: preserve the full path structure
            js_path = tsx_path.replace('_static/images/', '_static/js/').replace('.tsx', '.js')
        else:
            # For other paths, just replace extension but keep the directory structure
            js_path = os.path.splitext(tsx_path)[0] + '.js'

        # Create a unique ID for the component - PRESERVE THE ORIGINAL CASE
        component_name = os.path.basename(tsx_path).replace('.tsx', '')
        # Use component_name as-is without converting to lowercase
        div_id = f"react-{component_name}-{hash(tsx_path) % 10000}"
        error_id = f"error-{div_id}"
        wrapper_id = f"wrapper-{div_id}"

        # Create container styles based on options
        container_style = f"width: {width}; height: {height}; max-width: {max_width}; min-width: {min_width};"
        wrapper_style = ""

        if is_centered:
            wrapper_style += "display: flex; justify-content: center; align-items: center; width: 100%;"

        if align == 'left':
            wrapper_style += "display: flex; justify-content: flex-start; width: 100%;"
        elif align == 'right':
            wrapper_style += "display: flex; justify-content: flex-end; width: 100%;"

        # Add responsive styling if requested
        responsive_class = "react-component-responsive" if is_responsive else ""

        # KaTeX loading script - load KaTeX globally
        katex_script = ""
        if use_katex:
            katex_script = """
            // Only load KaTeX if it's not already loaded
            if (typeof window.katex === 'undefined') {
                // Load KaTeX CSS
                var katexCss = document.createElement('link');
                katexCss.rel = 'stylesheet';
                katexCss.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css';
                katexCss.integrity = 'sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntxDrHanlDqC0IIziTXcrXPnpVcVB8n2eHZ';
                katexCss.crossOrigin = 'anonymous';
                document.head.appendChild(katexCss);

                // Load KaTeX JS
                var katexScript = document.createElement('script');
                katexScript.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js';
                katexScript.integrity = 'sha384-cpW21h6RZv/phavutF+AuVYrr+dA8xD9zs6FwLpaCct6O9ctzYFfFr4dgmgccOTx';
                katexScript.crossOrigin = 'anonymous';
                katexScript.onload = function() {
                    // Once KaTeX is loaded, load the component
                    loadComponentScript();
                };
                document.head.appendChild(katexScript);
            } else {
                // KaTeX already loaded, proceed to load component
                loadComponentScript();
            }
            """
        else:
            # No KaTeX, just load the component directly
            katex_script = "loadComponentScript();"

        # Create HTML to load and mount the component with all styling options
        html = f"""
        <div id="{wrapper_id}" class="react-component-wrapper {custom_class}" style="{wrapper_style}">
            <div id="{div_id}" class="react-component-container {responsive_class}" style="{container_style}"></div>
            <div id="{error_id}" style="display: none; color: red; padding: 10px; border: 1px solid #ffcccc; margin-top: 10px; background-color: #fff8f8;"></div>
        </div>
        <script type="text/javascript">
            (function() {{
                // Self-contained function to avoid global scope pollution
                function showError(message) {{
                    // Display error in contained error div instead of modifying component container
                    var errorDiv = document.getElementById('{error_id}');
                    if (errorDiv) {{
                        errorDiv.innerHTML = '<strong>Error:</strong> ' + message;
                        errorDiv.style.display = 'block';
                        console.error(message);
                    }}
                }}

                function loadComponentScript() {{
                    try {{
                        // Only proceed if React is available
                        if (typeof React === 'undefined' || typeof ReactDOM === 'undefined') {{
                            showError('React or ReactDOM is not available');
                            return;
                        }}

                        // Load component script - using the exact filename and path
                        var script = document.createElement('script');
                        script.src = '{js_path}';
                        script.onerror = function(e) {{
                            showError('Failed to load component script: {js_path}');
                        }};
                        script.onload = function() {{
                            // Mount component with error handling - use the exact component name
                            try {{
                                if (window.{component_name}) {{
                                    ReactDOM.render(
                                        React.createElement(window.{component_name}, {{}}, null),
                                        document.getElementById('{div_id}')
                                    );
                                }} else {{
                                    showError('Component {component_name} not found in global scope');
                                }}
                            }} catch (error) {{
                                showError(error.message);
                            }}
                        }};
                        document.head.appendChild(script);
                    }} catch (error) {{
                        showError('Unexpected error: ' + error.message);
                    }}
                }}

                // Initialize component loading based on KaTeX option
                function initComponent() {{
                    {katex_script}
                }}

                // Initialize when DOM is ready
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', initComponent);
                }} else {{
                    initComponent();
                }}
            }})();
        </script>
        """

        raw_node = nodes.raw('', html, format='html')
        return [raw_node]


def setup(app):
    app.connect('builder-inited', convert_tsx_to_js)
    app.add_role('newconcept', NewConceptRole())
    app.add_role('refconcept', ref_newconcept_role)
    app.add_role('ul', underline_role)
    app.add_role('ub', underline_bold_role)
    app.add_css_file('custom.css')
    app.add_js_file("foldable_admonitions.js")
    app.add_directive('htmltable', HtmlTable)
    app.add_directive('note', CustomNoteDirective)
    app.add_directive('react-component', ReactComponentDirective)

    # Register the environment collector properly
    app.add_env_collector(NewConceptCollector)

    # Connect to the doctree-resolved event to process cross-references
    app.connect('doctree-resolved', process_newconcept_nodes)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
