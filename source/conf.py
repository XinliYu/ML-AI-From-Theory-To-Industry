from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxRole, SphinxDirective
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.directives.admonitions import BaseAdmonition
import re

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
    'sphinx.ext.viewcode',
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
    "colon_fence", # For code blocks with colons
    "tasklist",    # For task lists
    "deflist",     # For definition lists
    "html_admonition", # For admonitions
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
myst_parse_rst = True

templates_path = ['_templates']
html_js_files = ['js/mathjax-config.js']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_use_index = True
html_baseurl = 'https://xinliyu.github.io/ML-AI-From-Theory-To-Industry/'  # Important for GitHub Pages
html_use_relative_paths = True
numfig = True


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

def setup(app):
    app.add_role('newconcept', NewConceptRole())
    app.add_role('refconcept', ref_newconcept_role)
    app.add_role('ul', underline_role)
    app.add_role('ub', underline_bold_role)
    app.add_css_file('custom.css')
    app.add_js_file("foldable_admonitions.js")
    app.add_directive('note', CustomNoteDirective)
    
    # Register the environment collector properly
    app.add_env_collector(NewConceptCollector)
    
    # Connect to the doctree-resolved event to process cross-references
    app.connect('doctree-resolved', process_newconcept_nodes)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
