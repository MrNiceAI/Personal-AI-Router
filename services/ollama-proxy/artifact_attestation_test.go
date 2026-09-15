// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const (
	testArtifactA = "17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7"
	testArtifactB = "27052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7"
)

func attestedRequest(body string, digest string) *http.Request {
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	request.Header.Set(expectedArtifactSHA256Header, digest)
	return request
}

func writeArtifactInventory(w http.ResponseWriter, digest string) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = io.WriteString(w, `{"models":[{"name":"gpt-oss:20b","model":"gpt-oss:20b","digest":"`+digest+`"}]}`)
}

func TestHandleHTTP_ExactArtifactFiltersBeforeSelectionAndReplacesEngineEvidence(t *testing.T) {
	var mismatchInference atomic.Int32
	mismatch := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			_, _ = io.WriteString(w, `{"models":[{"name":"gpt-oss:20b","model":"gpt-oss:20b","digest":"`+testArtifactB+`"}]}`)
			return
		}
		mismatchInference.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer mismatch.Close()

	var matchInference atomic.Int32
	match := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" || r.URL.Path == "/api/ps" {
			if got := headerValues(r.Header, expectedArtifactSHA256Header); len(got) != 0 {
				t.Errorf("expected-artifact header leaked to inventory: %v", got)
			}
			writeArtifactInventory(w, testArtifactA)
			return
		}
		matchInference.Add(1)
		if got := headerValues(r.Header, expectedArtifactSHA256Header); len(got) != 0 {
			t.Errorf("expected-artifact header leaked to engine: %v", got)
		}
		w.Header().Add(servedArtifactSHA256Header, testArtifactB)
		w.Header().Add(servedArtifactSHA256Header, testArtifactB)
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer match.Close()

	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(t, "mismatch", mismatch.URL, "gpt-oss:20b"))
	discovery.AddManual(nodeForModel(t, "match", match.URL, "gpt-oss:20b"))
	proxy := testProxy(discovery, 11434)
	proxy.SetSelected("mismatch")
	recorder := httptest.NewRecorder()

	proxy.handleHTTP(recorder, attestedRequest(`{"model":"gpt-oss:20b"}`, testArtifactA))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if got := mismatchInference.Load(); got != 0 {
		t.Errorf("digest-mismatched selected node received %d inference requests, want 0", got)
	}
	if got := matchInference.Load(); got != 1 {
		t.Errorf("matching node received %d inference requests, want 1", got)
	}
	if got := headerValues(recorder.Header(), servedArtifactSHA256Header); len(got) != 1 || got[0] != testArtifactA {
		t.Fatalf("served artifact values = %v, want exactly [%s]", got, testArtifactA)
	}
}

func TestHandleHTTP_ExactArtifactFailoverKeepsFinalCandidateEvidence(t *testing.T) {
	server := func(status int, engineEvidence string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/tags" || r.URL.Path == "/api/ps" {
				writeArtifactInventory(w, testArtifactA)
				return
			}
			w.Header().Set(servedArtifactSHA256Header, engineEvidence)
			w.WriteHeader(status)
			_, _ = io.WriteString(w, `{"done":true}`)
		}))
	}
	busy := server(http.StatusServiceUnavailable, testArtifactB)
	defer busy.Close()
	good := server(http.StatusOK, testArtifactB)
	defer good.Close()

	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(t, "busy", busy.URL, "gpt-oss:20b"))
	discovery.AddManual(nodeForModel(t, "good", good.URL, "gpt-oss:20b"))
	proxy := testProxy(discovery, 11434)
	proxy.SetSelected("busy")
	recorder := httptest.NewRecorder()

	proxy.handleHTTP(recorder, attestedRequest(`{"model":"gpt-oss:20b"}`, testArtifactA))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 after failover; body=%s", recorder.Code, recorder.Body.String())
	}
	if got := headerValues(recorder.Header(), servedArtifactSHA256Header); len(got) != 1 || got[0] != testArtifactA {
		t.Fatalf("served artifact values = %v, want final candidate digest %s", got, testArtifactA)
	}
}

func TestHandleHTTP_ExactArtifactMismatchFailsClosed(t *testing.T) {
	var inference atomic.Int32
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			_, _ = io.WriteString(w, `{"models":[{"name":"gpt-oss:20b","digest":"`+testArtifactB+`"}]}`)
			return
		}
		inference.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer engine.Close()
	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(t, "engine", engine.URL, "gpt-oss:20b"))
	recorder := httptest.NewRecorder()

	testProxy(discovery, 11434).handleHTTP(recorder, attestedRequest(`{"model":"gpt-oss:20b"}`, testArtifactA))

	if recorder.Code != http.StatusPreconditionFailed {
		t.Fatalf("status = %d, want 412; body=%s", recorder.Code, recorder.Body.String())
	}
	if got := inference.Load(); got != 0 {
		t.Fatalf("mismatched engine received %d inference requests, want 0", got)
	}
	if got := headerValues(recorder.Header(), servedArtifactSHA256Header); len(got) != 0 {
		t.Fatalf("rejection carried served-artifact evidence: %v", got)
	}
}

func TestHandleHTTP_ExactArtifactInventoryUnavailableFailsClosed(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer engine.Close()
	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(t, "engine", engine.URL, "gpt-oss:20b"))
	recorder := httptest.NewRecorder()

	testProxy(discovery, 11434).handleHTTP(recorder, attestedRequest(`{"model":"gpt-oss:20b"}`, testArtifactA))

	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body=%s", recorder.Code, recorder.Body.String())
	}
}

func TestHandleHTTP_InvalidArtifactBindingRejectsLocally(t *testing.T) {
	tests := []struct {
		name   string
		path   string
		body   string
		values []string
	}{
		{name: "malformed", path: "/api/chat", body: `{"model":"gpt-oss:20b"}`, values: []string{"not-a-digest"}},
		{name: "uppercase", path: "/api/chat", body: `{"model":"gpt-oss:20b"}`, values: []string{strings.ToUpper(testArtifactA)}},
		{name: "duplicate", path: "/api/chat", body: `{"model":"gpt-oss:20b"}`, values: []string{testArtifactA, testArtifactA}},
		{name: "non-inference", path: "/api/version", body: `{}`, values: []string{testArtifactA}},
		{name: "missing-model", path: "/api/chat", body: `{}`, values: []string{testArtifactA}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body))
			request.Header[expectedArtifactSHA256Header] = test.values
			recorder := httptest.NewRecorder()
			testProxy(NewDiscovery(), 11434).handleHTTP(recorder, request)
			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
			}
		})
	}
}

func TestHandleHTTP_ExpectedArtifactTrailerIsRejected(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"gpt-oss:20b"}`))
	request.Trailer = http.Header{expectedArtifactSHA256Header: []string{testArtifactA}}
	recorder := httptest.NewRecorder()

	testProxy(NewDiscovery(), 11434).handleHTTP(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
	}
}

func TestHandleHTTP_InventoryRedirectIsNotFollowed(t *testing.T) {
	var redirected atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		redirected.Add(1)
		writeArtifactInventory(w, testArtifactA)
	}))
	defer target.Close()

	var inference atomic.Int32
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			http.Redirect(w, r, target.URL+"/api/tags", http.StatusTemporaryRedirect)
			return
		}
		inference.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer engine.Close()
	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(t, "engine", engine.URL, "gpt-oss:20b"))
	recorder := httptest.NewRecorder()

	testProxy(discovery, 11434).handleHTTP(recorder, attestedRequest(`{"model":"gpt-oss:20b"}`, testArtifactA))

	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body=%s", recorder.Code, recorder.Body.String())
	}
	if got := redirected.Load(); got != 0 {
		t.Fatalf("redirect target received %d requests, want 0", got)
	}
	if got := inference.Load(); got != 0 {
		t.Fatalf("redirecting engine received %d inference requests, want 0", got)
	}
}

func TestHandleHTTP_LoadedDigestMismatchFailsOverBeforeCommit(t *testing.T) {
	var staleInference atomic.Int32
	stale := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			writeArtifactInventory(w, testArtifactA)
		case "/api/ps":
			writeArtifactInventory(w, testArtifactB)
		default:
			staleInference.Add(1)
			_, _ = io.WriteString(w, `{"engine":"stale"}`)
		}
	}))
	defer stale.Close()

	var currentInference atomic.Int32
	current := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" || r.URL.Path == "/api/ps" {
			writeArtifactInventory(w, testArtifactA)
			return
		}
		currentInference.Add(1)
		_, _ = io.WriteString(w, `{"engine":"current"}`)
	}))
	defer current.Close()

	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(t, "stale", stale.URL, "gpt-oss:20b"))
	discovery.AddManual(nodeForModel(t, "current", current.URL, "gpt-oss:20b"))
	proxy := testProxy(discovery, 11434)
	proxy.SetSelected("stale")
	recorder := httptest.NewRecorder()

	proxy.handleHTTP(recorder, attestedRequest(`{"model":"gpt-oss:20b"}`, testArtifactA))

	if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), `"current"`) {
		t.Fatalf("final response = %d %s, want current candidate", recorder.Code, recorder.Body.String())
	}
	if staleInference.Load() != 1 || currentInference.Load() != 1 {
		t.Fatalf("inference counts stale=%d current=%d, want 1 each", staleInference.Load(), currentInference.Load())
	}
	if got := recorder.Header().Get(servedArtifactSHA256Header); got != testArtifactA {
		t.Fatalf("served artifact = %q, want %s", got, testArtifactA)
	}
}

func TestHandleHTTP_ArtifactInventoryConcurrencyIsBounded(t *testing.T) {
	var active atomic.Int32
	var maximum atomic.Int32
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			current := active.Add(1)
			defer active.Add(-1)
			for {
				observed := maximum.Load()
				if current <= observed || maximum.CompareAndSwap(observed, current) {
					break
				}
			}
			time.Sleep(25 * time.Millisecond)
			writeArtifactInventory(w, testArtifactA)
			return
		}
		if r.URL.Path == "/api/ps" {
			writeArtifactInventory(w, testArtifactA)
			return
		}
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer engine.Close()

	discovery := NewDiscovery()
	for i := 0; i < maxArtifactInventoryConcurrency+5; i++ {
		discovery.AddManual(nodeForModel(t, "engine-"+string(rune('a'+i)), engine.URL, "gpt-oss:20b"))
	}
	recorder := httptest.NewRecorder()
	testProxy(discovery, 11434).handleHTTP(recorder, attestedRequest(`{"model":"gpt-oss:20b"}`, testArtifactA))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if got := maximum.Load(); got > maxArtifactInventoryConcurrency {
		t.Fatalf("maximum concurrent inventory reads = %d, want <= %d", got, maxArtifactInventoryConcurrency)
	}
}

func TestHandleHTTP_UnboundRequestRetainsRoutingButStripsReservedEvidence(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Add(servedArtifactSHA256Header, testArtifactB)
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"done":true}`)
	}))
	defer engine.Close()
	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(t, "engine", engine.URL, "gpt-oss:20b"))
	recorder := httptest.NewRecorder()

	testProxy(discovery, 11434).handleHTTP(
		recorder,
		httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"gpt-oss:20b"}`)),
	)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", recorder.Code)
	}
	if got := headerValues(recorder.Header(), servedArtifactSHA256Header); len(got) != 0 {
		t.Fatalf("unbound response retained reserved engine evidence: %v", got)
	}
}

func TestHandleHTTP_StripsReservedResponseTrailer(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Add("Trailer", servedArtifactSHA256Header+", X-Engine-Trace")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"done":true}`)
		w.Header().Set(servedArtifactSHA256Header, testArtifactB)
		w.Header().Set("X-Engine-Trace", "preserved")
	}))
	defer engine.Close()
	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(t, "engine", engine.URL, "gpt-oss:20b"))
	recorder := httptest.NewRecorder()

	testProxy(discovery, 11434).handleHTTP(
		recorder,
		httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"gpt-oss:20b"}`)),
	)

	response := recorder.Result()
	defer response.Body.Close()
	_, _ = io.ReadAll(response.Body)
	if got := headerValues(response.Header, servedArtifactSHA256Header); len(got) != 0 {
		t.Fatalf("response header retained reserved evidence: %v", got)
	}
	if got := headerValues(response.Trailer, servedArtifactSHA256Header); len(got) != 0 {
		t.Fatalf("response trailer retained reserved evidence: %v", got)
	}
	if got := response.Trailer.Get("X-Engine-Trace"); got != "preserved" {
		t.Fatalf("unrelated trailer = %q, want preserved", got)
	}
}
