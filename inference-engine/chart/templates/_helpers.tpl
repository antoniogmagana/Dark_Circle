{{/* vim: set filetype=mustache: */}}
{{/*
Compute the full image reference for a given component.
Usage: {{ include "inference-engine.image" (dict "name" "discovery" "Values" .Values) }}
*/}}
{{- define "inference-engine.image" -}}
{{- printf "%s/%s:%s" .Values.images.registry .name .Values.images.tag -}}
{{- end -}}

{{/*
Render imagePullSecrets if any are configured.
Usage: {{ include "inference-engine.imagePullSecrets" . | nindent 6 }}
*/}}
{{- define "inference-engine.imagePullSecrets" -}}
{{- if .Values.images.pullSecrets -}}
imagePullSecrets:
{{- range .Values.images.pullSecrets }}
  - name: {{ .name }}
{{- end }}
{{- end -}}
{{- end -}}

{{/*
hostNetwork + dnsPolicy block emitted when ros2.hostNetwork is true.
Usage: {{ include "inference-engine.hostNetwork" . | nindent 6 }}
*/}}
{{- define "inference-engine.hostNetwork" -}}
{{- if .Values.ros2.hostNetwork -}}
hostNetwork: true
dnsPolicy: ClusterFirstWithHostNet
{{- end -}}
{{- end -}}

{{/*
FastDDS profile mount + env. The mount and volume always render so the
baked-in Ingestor template (which mounts unconditionally) doesn't fail
on a missing ConfigMap. The env var only renders when a profile is
configured, so an empty XML doesn't get pointed at.

These helpers accept either the chart root (.) — used by ingress-side
pods — or a dict {"root": ., "side": "egress"} — used by the egress
Deployment to pick up its optional separate profile. When side=egress
AND ros2.egress.fastddsProfile is set, the helpers point at the
fastdds-profiles-egress ConfigMap. Otherwise they fall through to the
shared fastdds-profiles ConfigMap.

Usage (ingress / default):
       {{ include "inference-engine.fastddsEnv" . | nindent 12 }}
       {{ include "inference-engine.fastddsMount" . | nindent 12 }}
       {{ include "inference-engine.fastddsVolume" . | nindent 6 }}
Usage (egress):
       {{ include "inference-engine.fastddsEnv" (dict "root" . "side" "egress") | nindent 12 }}
       (same dict for fastddsMount / fastddsVolume)
*/}}
{{- define "inference-engine.fastddsConfigMapName" -}}
{{- $values := .Values -}}
{{- $side := "" -}}
{{- if hasKey . "root" -}}
{{- $values = .root.Values -}}
{{- $side = .side | default "" -}}
{{- end -}}
{{- if and (eq $side "egress") $values.ros2.egress.fastddsProfile -}}
fastdds-profiles-egress
{{- else -}}
fastdds-profiles
{{- end -}}
{{- end -}}

{{- define "inference-engine.fastddsHasProfile" -}}
{{- $values := .Values -}}
{{- $side := "" -}}
{{- if hasKey . "root" -}}
{{- $values = .root.Values -}}
{{- $side = .side | default "" -}}
{{- end -}}
{{- if and (eq $side "egress") $values.ros2.egress.fastddsProfile -}}
true
{{- else if $values.ros2.fastddsProfile -}}
true
{{- end -}}
{{- end -}}

{{- define "inference-engine.fastddsEnv" -}}
{{- if eq (include "inference-engine.fastddsHasProfile" .) "true" -}}
- name: FASTRTPS_DEFAULT_PROFILES_FILE
  value: /etc/fastdds/fastdds.xml
{{- end -}}
{{- end -}}

{{- define "inference-engine.fastddsMount" -}}
- name: fastdds-profiles
  mountPath: /etc/fastdds
  readOnly: true
{{- end -}}

{{- define "inference-engine.fastddsVolume" -}}
- name: fastdds-profiles
  configMap:
    name: {{ include "inference-engine.fastddsConfigMapName" . }}
{{- end -}}
