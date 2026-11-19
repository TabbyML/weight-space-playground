// Product analytics via PostHog. The project key is a public, write-only key, so it is safe to
// ship in the bundle; VITE_POSTHOG_KEY / VITE_POSTHOG_HOST override it for other deployments
// and VITE_POSTHOG_DISABLED=true turns analytics off entirely.
//
// This is the same PostHog project as sketchagent.dev and zealt.dev: one project, with events
// separated by the page they came from ($current_url / $host). The site has no accounts, so
// events stay anonymous (no person profiles) and session recording is explicitly disabled.

const DEFAULT_KEY = 'phc_rqGVuiY7mN9yFMvW6HBzZWh7yY6U6m5ogGCr3Gtx8S2x';
const DEFAULT_HOST = 'https://us.i.posthog.com';
const DEFAULT_UI_HOST = 'https://us.posthog.com';

const env = import.meta.env;
const key = (env.VITE_POSTHOG_KEY || '').trim() || DEFAULT_KEY;
const host = (env.VITE_POSTHOG_HOST || '').trim() || DEFAULT_HOST;
const uiHost = (env.VITE_POSTHOG_UI_HOST || '').trim() || DEFAULT_UI_HOST;
const disabled = env.VITE_POSTHOG_DISABLED === 'true' || !key;

let starting = false;

/**
 * Load and initialize PostHog. Safe to call more than once; the import is async so it never
 * delays the first paint.
 */
export function initAnalytics() {
  if (disabled || typeof window === 'undefined' || starting) return;
  starting = true;
  import('posthog-js')
    .then(({ default: posthog }) => {
      posthog.init(key, {
        api_host: host,
        ui_host: uiHost,
        // No accounts, so anonymous events are enough (and cheaper than person profiles).
        person_profiles: 'identified_only',
        // The page is a single route, so PostHog's own pageview capture is enough.
        capture_pageview: true,
        capture_pageleave: true,
        autocapture: true,
        // Nothing here needs recording; keep it off as the app does.
        disable_session_recording: true,
        // Keep PostHog from injecting surveys/tours/site-app scripts from its CDN.
        disable_external_dependency_loading: true,
        disable_surveys: true,
        disable_product_tours: true,
        respect_dnt: true,
        sanitize_properties: sanitizeProperties,
      });
      posthog.register({ environment: env.MODE });
    })
    .catch((error) => {
      // Analytics must never break the page.
      console.warn('[analytics] PostHog failed to load', error);
    });
}

const URL_PROPERTIES = ['$current_url', '$initial_current_url', '$referrer', '$initial_referrer'];
const SENSITIVE_PARAMS = ['token', 'key', 'api_key', 'apikey', 'secret', 'code'];

/**
 * PostHog sends the current URL with every event. Strip fragments and credential-like query
 * parameters before anything leaves the browser.
 */
export function sanitizeProperties(properties) {
  for (const name of URL_PROPERTIES) {
    const value = properties[name];
    if (typeof value === 'string') properties[name] = stripSensitive(value);
  }
  return properties;
}

function stripSensitive(value) {
  try {
    const url = new URL(value);
    url.hash = '';
    for (const name of SENSITIVE_PARAMS) url.searchParams.delete(name);
    return url.toString();
  } catch {
    return value;
  }
}
