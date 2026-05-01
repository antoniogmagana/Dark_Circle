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
FastDDS profile mount + env, only emitted when ros2.fastddsProfile is non-empty.
Usage: {{ include "inference-engine.fastddsEnv" . | nindent 12 }}
       {{ include "inference-engine.fastddsMount" . | nindent 12 }}
       {{ include "inference-engine.fastddsVolume" . | nindent 6 }}
*/}}
{{- define "inference-engine.fastddsEnv" -}}
{{- if .Values.ros2.fastddsProfile -}}
- name: FASTRTPS_DEFAULT_PROFILES_FILE
  value: /etc/fastdds/fastdds.xml
{{- end -}}
{{- end -}}

{{- define "inference-engine.fastddsMount" -}}
{{- if .Values.ros2.fastddsProfile -}}
- name: fastdds-profiles
  mountPath: /etc/fastdds
  readOnly: true
{{- end -}}
{{- end -}}

{{- define "inference-engine.fastddsVolume" -}}
{{- if .Values.ros2.fastddsProfile -}}
- name: fastdds-profiles
  configMap:
    name: fastdds-profiles
{{- end -}}
{{- end -}}
