/*
 * support.js — minimal standalone runtime for this design-canvas (.dc.html) document.
 *
 * The page was authored against Anthropic's "design canvas" template dialect:
 *   - a <helmet> block whose children belong in <head>
 *   - a content tree using {{ expr }} interpolation, <sc-for>, <sc-if>, onClick="{{ fn }}"
 *   - a <script type="text/x-dc"> defining `class Component extends DCLogic`
 *
 * This file implements just enough of that runtime to render THIS document with no
 * external dependencies, so it can be hosted as a static site (e.g. GitHub Pages).
 */
(function () {
  "use strict";

  // Hide the raw template until we've rendered, so the user never sees `{{ ... }}`.
  try {
    var hideStyle = document.createElement("style");
    hideStyle.setAttribute("data-dc-runtime", "");
    hideStyle.textContent = "x-dc{visibility:hidden}";
    (document.head || document.documentElement).appendChild(hideStyle);
  } catch (e) {}

  // ---- expression + template helpers ---------------------------------------

  var exprCache = Object.create(null);
  function compileExpr(src) {
    if (exprCache[src]) return exprCache[src];
    var fn;
    try {
      // `with` lets bare identifiers and member access resolve against the scope.
      fn = new Function("$s", "with($s){return (" + src + ");}");
    } catch (e) {
      fn = function () { return undefined; };
    }
    exprCache[src] = fn;
    return fn;
  }
  function evalExpr(src, scope) {
    try { return compileExpr(src)(scope); } catch (e) { return undefined; }
  }

  // If a string is exactly a single `{{ ... }}` token, return the inner expression.
  function singleToken(value) {
    var m = /^\s*\{\{([\s\S]*?)\}\}\s*$/.exec(value);
    return m ? m[1].trim() : null;
  }

  function kebab(k) {
    return k.replace(/[A-Z]/g, function (c) { return "-" + c.toLowerCase(); });
  }
  function styleFromObject(obj) {
    var out = [];
    for (var k in obj) {
      if (!Object.prototype.hasOwnProperty.call(obj, k)) continue;
      var v = obj[k];
      if (v == null) continue;
      out.push(kebab(k) + ":" + v);
    }
    return out.join(";");
  }
  function tokenToCss(value) {
    // Resolve a token to a CSS fragment (object -> css string, else string).
    if (value && typeof value === "object") return styleFromObject(value);
    return value == null ? "" : "" + value;
  }

  // Replace every {{ ... }} in a string. `styleMode` serializes object results as CSS.
  function interpolate(str, scope, styleMode) {
    return str.replace(/\{\{([\s\S]*?)\}\}/g, function (_, expr) {
      var v = evalExpr(expr.trim(), scope);
      if (styleMode) return tokenToCss(v);
      return v == null ? "" : "" + v;
    });
  }

  // ---- template processing --------------------------------------------------

  var HINT_PREFIX = "hint-";

  // Process a list of source nodes against `scope`, returning live DOM nodes.
  function processNodes(nodeList, scope) {
    var out = [];
    for (var i = 0; i < nodeList.length; i++) {
      var produced = processNode(nodeList[i], scope);
      for (var j = 0; j < produced.length; j++) out.push(produced[j]);
    }
    return out;
  }

  function childScope(scope, key, value) {
    var s = Object.create(null);
    for (var k in scope) s[k] = scope[k];
    s[key] = value;
    return s;
  }

  function processNode(node, scope) {
    // Text
    if (node.nodeType === 3) {
      return [document.createTextNode(interpolate(node.nodeValue, scope, false))];
    }
    // Comments are dropped.
    if (node.nodeType === 8) return [];
    if (node.nodeType !== 1) return [];

    var tag = node.tagName.toLowerCase();

    if (tag === "sc-for") {
      var listExpr = singleToken(node.getAttribute("list") || "") ||
        (node.getAttribute("list") || "");
      var arr = evalExpr(listExpr, scope);
      if (!Array.isArray(arr)) arr = arr ? [].concat(arr) : [];
      var as = node.getAttribute("as") || "item";
      var rows = [];
      for (var n = 0; n < arr.length; n++) {
        var sc = childScope(scope, as, arr[n]);
        var produced = processNodes(node.childNodes, sc);
        for (var p = 0; p < produced.length; p++) rows.push(produced[p]);
      }
      return rows;
    }

    if (tag === "sc-if") {
      var valExpr = singleToken(node.getAttribute("value") || "") ||
        (node.getAttribute("value") || "");
      var truthy = evalExpr(valExpr, scope);
      return truthy ? processNodes(node.childNodes, scope) : [];
    }

    if (tag === "x-import") {
      // External component (e.g. image-slot.js) is not bundled — render a placeholder.
      var ph = document.createElement("div");
      var styleAttr = node.getAttribute("style") || "";
      var placeholder = node.getAttribute("placeholder") || "";
      ph.setAttribute(
        "style",
        interpolate(styleAttr, scope, true) +
          ";display:flex;align-items:center;justify-content:center;text-align:center;" +
          "border:1.5px dashed #D8D4CC;border-radius:12px;color:#A8A49B;" +
          "font:13px 'IBM Plex Sans',system-ui,sans-serif;padding:12px;box-sizing:border-box"
      );
      ph.textContent = placeholder;
      return [ph];
    }

    // Ordinary element.
    var el = document.createElement(tag);
    var attrs = node.attributes;
    for (var a = 0; a < attrs.length; a++) {
      var name = attrs[a].name;
      var value = attrs[a].value;
      var lname = name.toLowerCase();

      if (lname.indexOf(HINT_PREFIX) === 0) continue; // editor-only hints

      if (lname === "onclick") {
        var tok = singleToken(value);
        var fn = tok ? evalExpr(tok, scope) : null;
        if (typeof fn === "function") el.addEventListener("click", fn);
        continue;
      }

      if (lname === "style") {
        var stok = singleToken(value);
        if (stok) {
          el.setAttribute("style", tokenToCss(evalExpr(stok, scope)));
        } else {
          el.setAttribute("style", interpolate(value, scope, true));
        }
        continue;
      }

      var atok = singleToken(value);
      if (atok) {
        var av = evalExpr(atok, scope);
        if (typeof av === "function") continue; // unresolved handler-style binding
        el.setAttribute(name, av == null ? "" : "" + av);
      } else {
        el.setAttribute(name, interpolate(value, scope, false));
      }
    }

    var kids = processNodes(node.childNodes, scope);
    for (var c = 0; c < kids.length; c++) el.appendChild(kids[c]);
    return [el];
  }

  // ---- component base class -------------------------------------------------

  function DCLogic() {
    this.state = {};
    this.props = {};
  }
  DCLogic.prototype.setState = function (partial) {
    if (partial) for (var k in partial) this.state[k] = partial[k];
    if (this.__runtime) this.__runtime.scheduleRender();
  };
  DCLogic.prototype.componentDidMount = function () {};
  DCLogic.prototype.componentWillUnmount = function () {};

  // ---- runtime --------------------------------------------------------------

  function boot() {
    var xdc = document.querySelector("x-dc");
    if (!xdc) return;

    // 1. Hoist <helmet> children into <head>.
    var helmet = xdc.querySelector("helmet") || document.querySelector("helmet");
    if (helmet) {
      while (helmet.firstChild) document.head.appendChild(helmet.firstChild);
      helmet.parentNode && helmet.parentNode.removeChild(helmet);
    }

    // 2. Capture the content template (everything left inside <x-dc>).
    var tpl = document.createElement("template");
    tpl.innerHTML = xdc.innerHTML;
    var templateNodes = tpl.content.childNodes;

    // 3. Load the component definition + props.
    var scriptEl = document.querySelector('script[type="text/x-dc"][data-dc-script]') ||
      document.querySelector('script[type="text/x-dc"]');
    var props = {};
    var instance = null;

    if (scriptEl) {
      try {
        var raw = scriptEl.getAttribute("data-props");
        if (raw) {
          var spec = JSON.parse(raw);
          for (var key in spec) {
            if (key.charAt(0) === "$") continue; // metadata like $preview
            if (spec[key] && Object.prototype.hasOwnProperty.call(spec[key], "default")) {
              props[key] = spec[key].default;
            }
          }
        }
      } catch (e) { /* fall back to empty props */ }

      try {
        var factory = new Function("DCLogic", scriptEl.textContent + "\nreturn Component;");
        var Component = factory(DCLogic);
        instance = new Component();
      } catch (e) {
        // If the component fails to load, render the template statically (no bindings).
        console.error("[support.js] component init failed:", e);
      }
    }

    var reveal = function () {
      xdc.style.visibility = "visible";
      var hs = document.querySelector('style[data-dc-runtime]');
      hs && hs.parentNode && hs.parentNode.removeChild(hs);
    };

    if (!instance) {
      // Static fallback: render template with an empty scope.
      xdc.replaceChildren.apply(xdc, processNodes(templateNodes, Object.create(null)));
      reveal();
      return;
    }

    instance.props = props;

    var runtime = {
      pending: false,
      scheduleRender: function () {
        if (this.pending) return;
        this.pending = true;
        var self = this;
        (window.requestAnimationFrame || function (cb) { return setTimeout(cb, 16); })(function () {
          self.pending = false;
          render();
        });
      }
    };
    instance.__runtime = runtime;

    function render() {
      var values = instance.renderVals ? instance.renderVals() : {};
      var scope = Object.create(null);
      for (var k in values) scope[k] = values[k];
      var nodes = processNodes(templateNodes, scope);
      xdc.replaceChildren.apply(xdc, nodes);
    }

    render();
    reveal();
    try {
      if (typeof instance.componentDidMount === "function") instance.componentDidMount();
    } catch (e) { console.error("[support.js] componentDidMount failed:", e); }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
