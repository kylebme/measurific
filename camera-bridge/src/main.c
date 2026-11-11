#include <arv.h>
#include "focus.h"
#include <gio/gio.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef G_OS_WIN32
#include <fcntl.h>
#include <io.h>
#endif

#define PROTOCOL_VERSION 1
#define PACKET_JSON 1
#define PACKET_PREVIEW 2
#define PACKET_SNAPSHOT 3
#define PREVIEW_MAX_DIMENSION 2048
#define PREVIEW_INTERVAL_US 50000
#define STREAM_BUFFER_COUNT 8

typedef enum {
    PIXEL_UNSUPPORTED,
    PIXEL_MONO,
    PIXEL_RGB,
    PIXEL_BGR,
    PIXEL_RGBA,
    PIXEL_BGRA,
    PIXEL_BAYER_RG,
    PIXEL_BAYER_BG,
    PIXEL_BAYER_GR,
    PIXEL_BAYER_GB,
} PixelLayout;

typedef struct {
    PixelLayout layout;
    guint sensor_bits;
    guint bytes_per_pixel;
} PixelDescription;

/* Delivered image: the camera's own grayscale conversion, or one Bayer color site
 * at half resolution (never demosaiced). */
typedef enum {
    CHANNEL_CAMERA,
    CHANNEL_RED,
    CHANNEL_GREEN,
    CHANNEL_BLUE,
} OutputChannel;

static const gchar *channel_names[] = {"camera", "red", "green", "blue"};

typedef struct {
    guint8 *bytes;
    gsize size;
    gint width;
    gint height;
    gsize stride;
    guint64 sequence;
    guint64 timestamp_ns;
    PixelDescription pixel;
    OutputChannel channel;
    gint step;          /* 1, or 2 for a single Bayer color site */
    gint offset_x;      /* that site within the 2x2 tile */
    gint offset_y;
} RawFrame;

typedef struct {
    ArvCamera *camera;
    ArvStream *stream;
    GThread *thread;
    gint streaming;
    GMutex output_lock;
    GMutex frame_lock;
    RawFrame latest;
    gchar *device_id;
    gchar *vendor;
    gchar *model;
    gchar *serial;
    gchar *protocol;
    gchar *pixel_format;
    PixelDescription pixel;
    gint width;
    gint height;
    guint payload;
    OutputChannel channel;
    gint step;
    gint offset_x;
    gint offset_y;
    gboolean bayer_available;
    GPtrArray *saved_features;  /* camera color settings replaced for raw channels */
} Bridge;

static Bridge bridge;

static void put_u32_le(guint8 *target, guint32 value) {
    target[0] = value & 0xff;
    target[1] = (value >> 8) & 0xff;
    target[2] = (value >> 16) & 0xff;
    target[3] = (value >> 24) & 0xff;
}

static void write_packet(guint8 kind, const gchar *header, const guint8 *payload, gsize payload_size) {
    const gsize header_size = strlen(header);
    guint8 prefix[16] = {'I', 'C', 'A', 'M', PROTOCOL_VERSION, kind, 0, 0};
    if (header_size > G_MAXUINT32 || payload_size > G_MAXUINT32) return;
    put_u32_le(prefix + 8, (guint32)header_size);
    put_u32_le(prefix + 12, (guint32)payload_size);
    g_mutex_lock(&bridge.output_lock);
    fwrite(prefix, 1, sizeof(prefix), stdout);
    fwrite(header, 1, header_size, stdout);
    if (payload_size) fwrite(payload, 1, payload_size, stdout);
    fflush(stdout);
    g_mutex_unlock(&bridge.output_lock);
}

static gchar *json_quote(const gchar *value) {
    GString *out = g_string_new("\"");
    const guchar *cursor = (const guchar *)(value ? value : "");
    while (*cursor) {
        switch (*cursor) {
            case '\"': g_string_append(out, "\\\""); break;
            case '\\': g_string_append(out, "\\\\"); break;
            case '\b': g_string_append(out, "\\b"); break;
            case '\f': g_string_append(out, "\\f"); break;
            case '\n': g_string_append(out, "\\n"); break;
            case '\r': g_string_append(out, "\\r"); break;
            case '\t': g_string_append(out, "\\t"); break;
            default:
                if (*cursor < 0x20) g_string_append_printf(out, "\\u%04x", *cursor);
                else g_string_append_c(out, (gchar)*cursor);
        }
        cursor++;
    }
    g_string_append_c(out, '\"');
    return g_string_free(out, FALSE);
}

static void send_result(guint request_id, const gchar *result_json) {
    gchar *header = g_strdup_printf("{\"event\":\"response\",\"requestId\":%u,\"ok\":true,\"result\":%s}", request_id, result_json);
    write_packet(PACKET_JSON, header, NULL, 0);
    g_free(header);
}

static void send_error(guint request_id, const gchar *message) {
    gchar *quoted = json_quote(message);
    gchar *header = g_strdup_printf("{\"event\":\"response\",\"requestId\":%u,\"ok\":false,\"error\":%s}", request_id, quoted);
    write_packet(PACKET_JSON, header, NULL, 0);
    g_free(header);
    g_free(quoted);
}

static void send_gerror(guint request_id, const gchar *context, GError *error) {
    gchar *message = g_strdup_printf("%s: %s", context, error ? error->message : "unknown Aravis error");
    send_error(request_id, message);
    g_free(message);
    g_clear_error(&error);
}

static PixelDescription describe_pixel_format(const gchar *name) {
    PixelDescription result = {PIXEL_UNSUPPORTED, 0, 0};
    if (!name) return result;
    /* PFNC calls byte-aligned RGB layouts "Packed" too; only reject the
     * bit-packed Mono/Bayer forms that need a separate unpacker. */
    if (g_str_has_suffix(name, "p")) return result;
    if (g_str_has_prefix(name, "Mono")) result.layout = PIXEL_MONO;
    else if (g_str_has_prefix(name, "BayerRG")) result.layout = PIXEL_BAYER_RG;
    else if (g_str_has_prefix(name, "BayerBG")) result.layout = PIXEL_BAYER_BG;
    else if (g_str_has_prefix(name, "BayerGR")) result.layout = PIXEL_BAYER_GR;
    else if (g_str_has_prefix(name, "BayerGB")) result.layout = PIXEL_BAYER_GB;
    else if (!strcmp(name, "RGB8") || !strcmp(name, "RGB8Packed")) result.layout = PIXEL_RGB;
    else if (!strcmp(name, "BGR8") || !strcmp(name, "BGR8Packed")) result.layout = PIXEL_BGR;
    else if (!strcmp(name, "RGBA8") || !strcmp(name, "RGBA8Packed")) result.layout = PIXEL_RGBA;
    else if (!strcmp(name, "BGRA8") || !strcmp(name, "BGRA8Packed")) result.layout = PIXEL_BGRA;
    else return result;

    if (g_str_has_suffix(name, "Packed") &&
        (result.layout == PIXEL_MONO || result.layout >= PIXEL_BAYER_RG)) return (PixelDescription){PIXEL_UNSUPPORTED, 0, 0};

    const gchar *digit = name;
    while (*digit && !g_ascii_isdigit(*digit)) digit++;
    result.sensor_bits = *digit ? (guint)g_ascii_strtoull(digit, NULL, 10) : 8;
    if (result.layout == PIXEL_RGB || result.layout == PIXEL_BGR) result.bytes_per_pixel = 3;
    else if (result.layout == PIXEL_RGBA || result.layout == PIXEL_BGRA) result.bytes_per_pixel = 4;
    else result.bytes_per_pixel = result.sensor_bits > 8 ? 2 : 1;
    if (result.sensor_bits < 8 || result.sensor_bits > 16) result.layout = PIXEL_UNSUPPORTED;
    return result;
}

static gboolean format_is_available(const gchar *candidate, const gchar **formats, guint count) {
    for (guint i = 0; i < count; i++) if (!g_strcmp0(candidate, formats[i])) return TRUE;
    return FALSE;
}

static gchar *choose_pixel_format(ArvCamera *camera, GError **error) {
    static const gchar *preference[] = {
        "Mono16", "Mono14", "Mono12", "Mono10", "Mono8",
        "BayerRG16", "BayerBG16", "BayerGR16", "BayerGB16",
        "BayerRG14", "BayerBG14", "BayerGR14", "BayerGB14",
        "BayerRG12", "BayerBG12", "BayerGR12", "BayerGB12",
        "BayerRG10", "BayerBG10", "BayerGR10", "BayerGB10",
        "RGB8", "BGR8", "RGBA8", "BGRA8",
        "RGB8Packed", "BGR8Packed", "RGBA8Packed", "BGRA8Packed",
        "BayerRG8", "BayerBG8", "BayerGR8", "BayerGB8",
    };
    guint count = 0;
    const gchar **formats = arv_camera_dup_available_pixel_formats_as_strings(camera, &count, error);
    if (*error) return NULL;
    const gchar *selected = NULL;
    for (guint i = 0; i < G_N_ELEMENTS(preference) && !selected; i++) {
        if (format_is_available(preference[i], formats, count)) selected = preference[i];
    }
    if (!selected) {
        for (guint i = 0; i < count; i++) {
            if (describe_pixel_format(formats[i]).layout != PIXEL_UNSUPPORTED) { selected = formats[i]; break; }
        }
    }
    gchar *copy = selected ? g_strdup(selected) : NULL;
    g_free(formats);
    if (!copy) g_set_error(error, G_IO_ERROR, G_IO_ERROR_NOT_SUPPORTED, "Camera has no supported unpacked Mono, RGB, or Bayer pixel format");
    return copy;
}

/* Deepest unpacked Bayer format, or NULL for cameras without raw color output. */
static gchar *choose_bayer_format(ArvCamera *camera) {
    static const gchar *depths[] = {"16", "14", "12", "10", "8"};
    static const gchar *patterns[] = {"BayerRG", "BayerBG", "BayerGR", "BayerGB"};
    guint count = 0;
    const gchar **formats = arv_camera_dup_available_pixel_formats_as_strings(camera, &count, NULL);
    gchar *selected = NULL;
    for (guint d = 0; d < G_N_ELEMENTS(depths) && !selected && formats; d++) {
        for (guint p = 0; p < G_N_ELEMENTS(patterns) && !selected; p++) {
            gchar *name = g_strconcat(patterns[p], depths[d], NULL);
            if (format_is_available(name, formats, count)) selected = name;
            else g_free(name);
        }
    }
    g_free(formats);
    return selected;
}

static guint16 read_sample(const guint8 *row, gint x, const PixelDescription *pixel) {
    if (pixel->bytes_per_pixel == 1) return row[x];
    const guint8 *p = row + 2 * x;
    guint16 value = (guint16)(p[0] | ((guint16)p[1] << 8));
    return pixel->sensor_bits < 16 ? (guint16)(value << (16 - pixel->sensor_bits)) : value;
}

static gint bayer_color(PixelLayout layout, gint x, gint y) {
    const gboolean even_x = !(x & 1), even_y = !(y & 1);
    if (layout == PIXEL_BAYER_RG) return even_y ? (even_x ? 0 : 1) : (even_x ? 1 : 2);
    if (layout == PIXEL_BAYER_BG) return even_y ? (even_x ? 2 : 1) : (even_x ? 1 : 0);
    if (layout == PIXEL_BAYER_GR) return even_y ? (even_x ? 1 : 0) : (even_x ? 2 : 1);
    return even_y ? (even_x ? 1 : 2) : (even_x ? 0 : 1);
}

/* Site of a color within the 2x2 Bayer tile. Green uses the green in red's row, so
 * every channel is one site on a regular 2-pixel grid. */
static void channel_site(PixelLayout layout, OutputChannel channel, gint *offset_x, gint *offset_y) {
    const gint wanted = channel == CHANNEL_RED ? 0 : channel == CHANNEL_GREEN ? 1 : 2;
    gint red_row = 0;
    for (gint y = 0; y < 2; y++) for (gint x = 0; x < 2; x++) if (bayer_color(layout, x, y) == 0) red_row = y;
    for (gint y = 0; y < 2; y++) {
        for (gint x = 0; x < 2; x++) {
            if (bayer_color(layout, x, y) == wanted && (wanted != 1 || y == red_row)) {
                *offset_x = x; *offset_y = y; return;
            }
        }
    }
}

static guint16 gray_sample(const RawFrame *frame, gint x, gint y) {
    const guint8 *row = frame->bytes + (gsize)y * frame->stride;
    if (frame->pixel.layout == PIXEL_MONO) return read_sample(row, x, &frame->pixel);
    if (frame->pixel.layout == PIXEL_RGB || frame->pixel.layout == PIXEL_BGR || frame->pixel.layout == PIXEL_RGBA || frame->pixel.layout == PIXEL_BGRA) {
        const guint channels = frame->pixel.bytes_per_pixel;
        const guint8 *p = row + channels * x;
        const gboolean bgr = frame->pixel.layout == PIXEL_BGR || frame->pixel.layout == PIXEL_BGRA;
        const guint r = p[bgr ? 2 : 0], g = p[1], b = p[bgr ? 0 : 2];
        return (guint16)((77 * r + 150 * g + 29 * b) >> 8);
    }
    guint64 sum[3] = {0, 0, 0};
    guint count[3] = {0, 0, 0};
    for (gint dy = -1; dy <= 1; dy++) {
        const gint sy = CLAMP(y + dy, 0, frame->height - 1);
        const guint8 *source_row = frame->bytes + (gsize)sy * frame->stride;
        for (gint dx = -1; dx <= 1; dx++) {
            const gint sx = CLAMP(x + dx, 0, frame->width - 1);
            const gint color = bayer_color(frame->pixel.layout, sx, sy);
            sum[color] += read_sample(source_row, sx, &frame->pixel);
            count[color]++;
        }
    }
    const guint64 center = read_sample(row, x, &frame->pixel);
    const guint64 r = count[0] ? sum[0] / count[0] : center;
    const guint64 g = count[1] ? sum[1] / count[1] : center;
    const guint64 b = count[2] ? sum[2] / count[2] : center;
    return (guint16)((77 * r + 150 * g + 29 * b) >> 8);
}

static gint output_width(const RawFrame *frame) {
    return frame->step == 2 ? (frame->width - frame->offset_x + 1) / 2 : frame->width;
}

static gint output_height(const RawFrame *frame) {
    return frame->step == 2 ? (frame->height - frame->offset_y + 1) / 2 : frame->height;
}

/* One delivered sample: a single Bayer site, or the camera-mode grayscale value. */
static guint16 output_sample(const RawFrame *frame, gint x, gint y) {
    if (frame->step == 2) {
        const guint8 *row = frame->bytes + (gsize)(2 * y + frame->offset_y) * frame->stride;
        return read_sample(row, 2 * x + frame->offset_x, &frame->pixel);
    }
    return gray_sample(frame, x, y);
}

static guint8 *convert_gray(const RawFrame *frame, gboolean preview, gint *out_width, gint *out_height, guint *storage_bits, gsize *out_size) {
    const gint source_width = output_width(frame), source_height = output_height(frame);
    double scale = 1.0;
    if (preview && MAX(source_width, source_height) > PREVIEW_MAX_DIMENSION) scale = (double)PREVIEW_MAX_DIMENSION / MAX(source_width, source_height);
    *out_width = MAX(1, (gint)floor(source_width * scale));
    *out_height = MAX(1, (gint)floor(source_height * scale));
    *storage_bits = preview || frame->pixel.sensor_bits <= 8 ? 8 : 16;
    const guint bytes_per_output = *storage_bits / 8;
    *out_size = (gsize)*out_width * *out_height * bytes_per_output;
    guint8 *output = g_malloc(*out_size);
    for (gint y = 0; y < *out_height; y++) {
        const gint source_y = MIN(source_height - 1, (gint)((gint64)y * source_height / *out_height));
        for (gint x = 0; x < *out_width; x++) {
            const gint source_x = MIN(source_width - 1, (gint)((gint64)x * source_width / *out_width));
            guint16 value = output_sample(frame, source_x, source_y);
            if (*storage_bits == 8) {
                output[(gsize)y * *out_width + x] = frame->pixel.sensor_bits <= 8 ? (guint8)value : (guint8)(value >> 8);
            } else {
                const gsize offset = 2 * ((gsize)y * *out_width + x);
                output[offset] = value & 0xff;
                output[offset + 1] = value >> 8;
            }
        }
    }
    return output;
}

/* fullWidth/fullHeight are the delivered (channel) frame; sensor pixel of delivered
 * pixel (x, y) is (sampleStep * x + sampleOffsetX, sampleStep * y + sampleOffsetY). */
static gchar *frame_header(const RawFrame *frame, gint width, gint height, guint storage_bits, guint request_id) {
    gchar *format = json_quote(bridge.pixel_format);
    gchar *header = g_strdup_printf(
        "{\"%s\":%u,\"width\":%d,\"height\":%d,\"fullWidth\":%d,\"fullHeight\":%d,"
        "\"sensorWidth\":%d,\"sensorHeight\":%d,\"channel\":\"%s\",\"sampleStep\":%d,\"sampleOffsetX\":%d,\"sampleOffsetY\":%d,"
        "\"storageBits\":%u,\"sensorBits\":%u,\"pixelFormat\":%s,\"sequence\":%" G_GUINT64_FORMAT ",\"timestampNs\":%" G_GUINT64_FORMAT "}",
        request_id ? "requestId" : "stream", request_id, width, height, output_width(frame), output_height(frame),
        frame->width, frame->height, channel_names[frame->channel], frame->step, frame->offset_x, frame->offset_y,
        storage_bits, frame->pixel.sensor_bits, format, frame->sequence, frame->timestamp_ns);
    g_free(format);
    return header;
}

static void clear_raw_frame(RawFrame *frame) {
    g_free(frame->bytes);
    memset(frame, 0, sizeof(*frame));
}

static void store_raw_frame(ArvBuffer *buffer) {
    gsize size = 0;
    const guint8 *data = arv_buffer_get_part_data(buffer, 0, &size);
    const gint width = arv_buffer_get_image_width(buffer);
    const gint height = arv_buffer_get_image_height(buffer);
    if (!data || width <= 0 || height <= 0) return;
    const gsize minimum_stride = (gsize)width * bridge.pixel.bytes_per_pixel;
    gint x_padding = 0;
    arv_buffer_get_image_padding(buffer, &x_padding, NULL);
    const gsize stride = minimum_stride + MAX(0, x_padding);
    const gsize frame_size = stride * (gsize)height;
    if (stride < minimum_stride || size < frame_size) return;
    g_mutex_lock(&bridge.frame_lock);
    if (bridge.latest.size != frame_size) {
        bridge.latest.bytes = g_realloc(bridge.latest.bytes, frame_size);
        bridge.latest.size = frame_size;
    }
    memcpy(bridge.latest.bytes, data, bridge.latest.size);
    bridge.latest.width = width;
    bridge.latest.height = height;
    bridge.latest.stride = stride;
    bridge.latest.pixel = bridge.pixel;
    bridge.latest.channel = bridge.channel;
    bridge.latest.step = bridge.step;
    bridge.latest.offset_x = bridge.offset_x;
    bridge.latest.offset_y = bridge.offset_y;
    bridge.latest.sequence = arv_buffer_get_frame_id(buffer);
    bridge.latest.timestamp_ns = arv_buffer_get_timestamp(buffer);
    g_mutex_unlock(&bridge.frame_lock);
}

static gpointer stream_thread(gpointer unused) {
    (void)unused;
    gint64 last_preview = 0;
    while (g_atomic_int_get(&bridge.streaming)) {
        ArvBuffer *buffer = arv_stream_timeout_pop_buffer(bridge.stream, 100000);
        if (!buffer) continue;
        if (arv_buffer_get_status(buffer) == ARV_BUFFER_STATUS_SUCCESS) {
            const gint64 now = g_get_monotonic_time();
            if (now - last_preview >= PREVIEW_INTERVAL_US) {
                store_raw_frame(buffer);
                g_mutex_lock(&bridge.frame_lock);
                gint width, height;
                guint storage_bits;
                gsize output_size;
                guint8 *output = convert_gray(&bridge.latest, TRUE, &width, &height, &storage_bits, &output_size);
                gchar *header = frame_header(&bridge.latest, width, height, storage_bits, 0);
                g_mutex_unlock(&bridge.frame_lock);
                write_packet(PACKET_PREVIEW, header, output, output_size);
                g_free(header);
                g_free(output);
                last_preview = now;
            }
        }
        arv_stream_push_buffer(bridge.stream, buffer);
    }
    return NULL;
}

static void stop_stream(void) {
    if (!bridge.thread) return;
    g_atomic_int_set(&bridge.streaming, FALSE);
    g_thread_join(bridge.thread);
    bridge.thread = NULL;
    if (bridge.camera) arv_camera_stop_acquisition(bridge.camera, NULL);
}

/* Raw channels need the sensor's linear samples. Where a camera applies color
 * processing to Bayer output too (white balance, presets, color space, gamma, LUT),
 * these optional features are switched off; the originals are restored for camera
 * output and on close. Features a camera lacks are skipped. */
static const struct { const gchar *name; const gchar *selector; const gchar *raw; } raw_settings[] = {
    {"BalanceWhiteAuto", NULL, "Off"},
    {"BslLightSourcePreset", NULL, "Off"},
    {"BslColorSpace", NULL, "Off"},
    {"LUTEnable", NULL, "false"},
    {"GammaEnable", NULL, "false"},
    {"Gamma", NULL, "1"},
    {"BalanceRatio", "Red", "1"},
    {"BalanceRatio", "Green", "1"},
    {"BalanceRatio", "Blue", "1"},
};

typedef struct { gchar *name; gchar *selector; gchar *value; } SavedFeature;

static void saved_feature_free(gpointer data) {
    SavedFeature *saved = data;
    g_free(saved->name); g_free(saved->selector); g_free(saved->value); g_free(saved);
}

static ArvGcFeatureNode *feature_node(const gchar *name) {
    ArvGcNode *node = arv_device_get_feature(arv_camera_get_device(bridge.camera), name);
    if (!ARV_IS_GC_FEATURE_NODE(node)) return NULL;
    ArvGcFeatureNode *feature = ARV_GC_FEATURE_NODE(node);
    if (!arv_gc_feature_node_is_implemented(feature, NULL) || !arv_gc_feature_node_is_available(feature, NULL)) return NULL;
    return feature;
}

/* The feature, with its selector set first; NULL if either is missing. */
static ArvGcFeatureNode *selected_feature(const gchar *name, const gchar *selector) {
    if (selector) {
        gchar *selector_name = g_strconcat(name, "Selector", NULL);
        ArvGcFeatureNode *node = feature_node(selector_name);
        GError *error = NULL;
        if (node) arv_gc_feature_node_set_value_from_string(node, selector, &error);
        g_free(selector_name);
        if (!node || error) { g_clear_error(&error); return NULL; }
    }
    return feature_node(name);
}

static void use_raw_samples(void) {
    if (!bridge.saved_features) {
        bridge.saved_features = g_ptr_array_new_with_free_func(saved_feature_free);
        for (guint i = 0; i < G_N_ELEMENTS(raw_settings); i++) {
            ArvGcFeatureNode *feature = selected_feature(raw_settings[i].name, raw_settings[i].selector);
            const gchar *value = feature ? arv_gc_feature_node_get_value_as_string(feature, NULL) : NULL;
            if (!value) continue;
            SavedFeature *saved = g_new0(SavedFeature, 1);
            saved->name = g_strdup(raw_settings[i].name);
            saved->selector = g_strdup(raw_settings[i].selector);
            saved->value = g_strdup(value);
            g_ptr_array_add(bridge.saved_features, saved);
        }
    }
    for (guint i = 0; i < G_N_ELEMENTS(raw_settings); i++) {
        ArvGcFeatureNode *feature = selected_feature(raw_settings[i].name, raw_settings[i].selector);
        if (feature) arv_gc_feature_node_set_value_from_string(feature, raw_settings[i].raw, NULL);
    }
}

static void restore_camera_processing(void) {
    if (!bridge.saved_features) return;
    /* Undo in reverse order, since a feature can depend on a later one (a
     * Basler light-source preset is only writable with its color space on).
     * Presets rewrite the balance ratios, so those come next-to-last, and
     * automatic white balance, which would overwrite them, last. */
    for (gint pass = 0; pass < 3; pass++) {
        for (gint i = (gint)bridge.saved_features->len - 1; i >= 0; i--) {
            SavedFeature *saved = g_ptr_array_index(bridge.saved_features, i);
            const gint group = !strcmp(saved->name, "BalanceWhiteAuto") ? 2 : !strcmp(saved->name, "BalanceRatio") ? 1 : 0;
            if (group != pass) continue;
            ArvGcFeatureNode *feature = selected_feature(saved->name, saved->selector);
            if (feature) arv_gc_feature_node_set_value_from_string(feature, saved->value, NULL);
        }
    }
    g_clear_pointer(&bridge.saved_features, g_ptr_array_unref);
}

static gboolean start_stream(GError **error) {
    arv_camera_set_acquisition_mode(bridge.camera, ARV_ACQUISITION_MODE_CONTINUOUS, error);
    if (!*error) arv_camera_start_acquisition(bridge.camera, error);
    if (*error) return FALSE;
    g_atomic_int_set(&bridge.streaming, TRUE);
    bridge.thread = g_thread_new("camera-stream", stream_thread, NULL);
    return TRUE;
}

/* Select the pixel format and processing for a channel, rebuilding the stream
 * buffers for the new payload. A running stream is restarted. */
static gboolean configure_output(OutputChannel channel, GError **error) {
    gchar *format = NULL;
    if (channel == CHANNEL_CAMERA) {
        format = choose_pixel_format(bridge.camera, error);
        if (!format) return FALSE;
    } else {
        format = choose_bayer_format(bridge.camera);
        if (!format) {
            g_set_error(error, G_IO_ERROR, G_IO_ERROR_NOT_SUPPORTED,
                        "This camera does not deliver raw Bayer data; single-channel output needs a color camera");
            return FALSE;
        }
    }
    const gboolean was_streaming = bridge.thread != NULL;
    stop_stream();
    arv_camera_set_pixel_format_from_string(bridge.camera, format, error);
    if (!*error) {
        if (channel == CHANNEL_CAMERA) restore_camera_processing();
        else use_raw_samples();
        bridge.payload = arv_camera_get_payload(bridge.camera, error);
    }
    if (!*error) {
        g_clear_object(&bridge.stream);
        bridge.stream = arv_camera_create_stream(bridge.camera, NULL, NULL, error);
    }
    if (*error || !bridge.stream) {
        g_free(format);
        if (!*error) g_set_error(error, G_IO_ERROR, G_IO_ERROR_FAILED, "Unable to create camera stream");
        return FALSE;
    }
    for (guint i = 0; i < STREAM_BUFFER_COUNT; i++) arv_stream_push_buffer(bridge.stream, arv_buffer_new(bridge.payload, NULL));
    g_free(bridge.pixel_format);
    bridge.pixel_format = format;
    bridge.pixel = describe_pixel_format(format);
    bridge.channel = channel;
    bridge.step = channel == CHANNEL_CAMERA ? 1 : 2;
    bridge.offset_x = bridge.offset_y = 0;
    if (channel != CHANNEL_CAMERA) channel_site(bridge.pixel.layout, channel, &bridge.offset_x, &bridge.offset_y);
    g_mutex_lock(&bridge.frame_lock);
    clear_raw_frame(&bridge.latest);
    g_mutex_unlock(&bridge.frame_lock);
    return was_streaming ? start_stream(error) : TRUE;
}

static void close_camera(void) {
    stop_stream();
    if (bridge.camera) restore_camera_processing();
    if (bridge.stream) g_object_unref(bridge.stream);
    if (bridge.camera) g_object_unref(bridge.camera);
    bridge.stream = NULL;
    bridge.camera = NULL;
    clear_raw_frame(&bridge.latest);
    g_clear_pointer(&bridge.device_id, g_free);
    g_clear_pointer(&bridge.vendor, g_free);
    g_clear_pointer(&bridge.model, g_free);
    g_clear_pointer(&bridge.serial, g_free);
    g_clear_pointer(&bridge.protocol, g_free);
    g_clear_pointer(&bridge.pixel_format, g_free);
    bridge.width = bridge.height = 0;
    bridge.channel = CHANNEL_CAMERA;
    bridge.step = 1;
    bridge.offset_x = bridge.offset_y = 0;
    bridge.bayer_available = FALSE;
}

static gchar *controls_json(void) {
    gboolean exposure = FALSE, exposure_auto = FALSE, gain = FALSE, gain_auto = FALSE;
    double exposure_value = 0, exposure_min = 0, exposure_max = 0;
    double gain_value = 0, gain_min = 0, gain_max = 0;
    if (bridge.camera) {
        exposure = arv_camera_is_exposure_time_available(bridge.camera, NULL);
        exposure_auto = exposure && arv_camera_is_exposure_auto_available(bridge.camera, NULL);
        gain = arv_camera_is_gain_available(bridge.camera, NULL);
        gain_auto = gain && arv_camera_is_gain_auto_available(bridge.camera, NULL);
        if (exposure) {
            exposure_value = arv_camera_get_exposure_time(bridge.camera, NULL);
            arv_camera_get_exposure_time_bounds(bridge.camera, &exposure_min, &exposure_max, NULL);
        }
        if (gain) {
            gain_value = arv_camera_get_gain(bridge.camera, NULL);
            arv_camera_get_gain_bounds(bridge.camera, &gain_min, &gain_max, NULL);
        }
    }
    const gboolean exposure_is_auto = exposure_auto && arv_camera_get_exposure_time_auto(bridge.camera, NULL) != ARV_AUTO_OFF;
    const gboolean gain_is_auto = gain_auto && arv_camera_get_gain_auto(bridge.camera, NULL) != ARV_AUTO_OFF;
    return g_strdup_printf(
        "{\"exposure\":{\"available\":%s,\"autoAvailable\":%s,\"auto\":%s,\"value\":%.10g,\"min\":%.10g,\"max\":%.10g},"
        "\"gain\":{\"available\":%s,\"autoAvailable\":%s,\"auto\":%s,\"value\":%.10g,\"min\":%.10g,\"max\":%.10g}}",
        exposure ? "true" : "false", exposure_auto ? "true" : "false", exposure_is_auto ? "true" : "false", exposure_value, exposure_min, exposure_max,
        gain ? "true" : "false", gain_auto ? "true" : "false", gain_is_auto ? "true" : "false", gain_value, gain_min, gain_max);
}

static gchar *camera_json(void) {
    gchar *id = json_quote(bridge.device_id), *vendor = json_quote(bridge.vendor), *model = json_quote(bridge.model);
    gchar *serial = json_quote(bridge.serial), *protocol = json_quote(bridge.protocol), *format = json_quote(bridge.pixel_format);
    const gint width = bridge.step == 2 ? (bridge.width - bridge.offset_x + 1) / 2 : bridge.width;
    const gint height = bridge.step == 2 ? (bridge.height - bridge.offset_y + 1) / 2 : bridge.height;
    gchar *json = g_strdup_printf(
        "{\"id\":%s,\"vendor\":%s,\"model\":%s,\"serial\":%s,\"protocol\":%s,\"pixelFormat\":%s,\"sensorBits\":%u,"
        "\"width\":%d,\"height\":%d,\"sensorWidth\":%d,\"sensorHeight\":%d,\"channel\":\"%s\",\"channels\":%s}",
        id, vendor, model, serial, protocol, format, bridge.pixel.sensor_bits, width, height, bridge.width, bridge.height,
        channel_names[bridge.channel], bridge.bayer_available ? "[\"camera\",\"red\",\"green\",\"blue\"]" : "[\"camera\"]");
    g_free(id); g_free(vendor); g_free(model); g_free(serial); g_free(protocol); g_free(format);
    return json;
}

static void command_discover(guint request_id) {
    arv_update_device_list();
    GString *devices = g_string_new("{\"devices\":[");
    const guint count = arv_get_n_devices();
    for (guint i = 0; i < count; i++) {
        gchar *id = json_quote(arv_get_device_id(i));
        gchar *vendor = json_quote(arv_get_device_vendor(i));
        gchar *model = json_quote(arv_get_device_model(i));
        gchar *serial = json_quote(arv_get_device_serial_nbr(i));
        gchar *protocol = json_quote(arv_get_device_protocol(i));
        if (i) g_string_append_c(devices, ',');
        g_string_append_printf(devices, "{\"id\":%s,\"vendor\":%s,\"model\":%s,\"serial\":%s,\"protocol\":%s}", id, vendor, model, serial, protocol);
        g_free(id); g_free(vendor); g_free(model); g_free(serial); g_free(protocol);
    }
    g_string_append(devices, "]}");
    send_result(request_id, devices->str);
    g_string_free(devices, TRUE);
}

static const gchar *protocol_for_device(const gchar *device_id) {
    const guint count = arv_get_n_devices();
    for (guint i = 0; i < count; i++) if (!g_strcmp0(device_id, arv_get_device_id(i))) return arv_get_device_protocol(i);
    return "GenICam";
}

static void command_open(guint request_id, const gchar *encoded_id) {
    gchar *device_id = g_uri_unescape_string(encoded_id, NULL);
    if (!device_id) { send_error(request_id, "Invalid camera identifier"); return; }
    close_camera();
    GError *error = NULL;
    bridge.camera = arv_camera_new(device_id, &error);
    if (!bridge.camera) { send_gerror(request_id, "Unable to open camera", error); g_free(device_id); return; }
    gchar *bayer = choose_bayer_format(bridge.camera);
    bridge.bayer_available = bayer != NULL;
    g_free(bayer);
    arv_camera_get_region(bridge.camera, NULL, NULL, &bridge.width, &bridge.height, &error);
    if (error) { send_gerror(request_id, "Unable to read camera region", error); close_camera(); g_free(device_id); return; }
    /* Opens with the camera's own output; the app then selects its channel. */
    if (!configure_output(CHANNEL_CAMERA, &error)) { send_gerror(request_id, "Unable to configure camera output", error); close_camera(); g_free(device_id); return; }
    bridge.device_id = g_strdup(device_id);
    bridge.vendor = g_strdup(arv_camera_get_vendor_name(bridge.camera, NULL));
    bridge.model = g_strdup(arv_camera_get_model_name(bridge.camera, NULL));
    bridge.serial = g_strdup(arv_camera_get_device_serial_number(bridge.camera, NULL));
    bridge.protocol = g_strdup(protocol_for_device(device_id));
    gchar *camera = camera_json(), *controls = controls_json();
    gchar *result = g_strdup_printf("{\"camera\":%s,\"controls\":%s}", camera, controls);
    send_result(request_id, result);
    g_free(result); g_free(camera); g_free(controls); g_free(device_id);
}

static void command_start(guint request_id) {
    if (!bridge.camera || !bridge.stream) { send_error(request_id, "Open a camera before starting acquisition"); return; }
    if (!bridge.thread) {
        GError *error = NULL;
        if (!start_stream(&error)) { send_gerror(request_id, "Unable to start acquisition", error); return; }
    }
    gchar *camera = camera_json(), *controls = controls_json();
    gchar *result = g_strdup_printf("{\"camera\":%s,\"controls\":%s}", camera, controls);
    send_result(request_id, result);
    g_free(result); g_free(camera); g_free(controls);
}

static void command_pause(guint request_id) {
    stop_stream();
    g_mutex_lock(&bridge.frame_lock);
    if (!bridge.latest.bytes) {
        g_mutex_unlock(&bridge.frame_lock);
        send_error(request_id, "No complete camera frame was received");
        return;
    }
    gint width, height;
    guint storage_bits;
    gsize output_size;
    guint8 *output = convert_gray(&bridge.latest, FALSE, &width, &height, &storage_bits, &output_size);
    gchar *header = frame_header(&bridge.latest, width, height, storage_bits, request_id);
    g_mutex_unlock(&bridge.frame_lock);
    write_packet(PACKET_SNAPSHOT, header, output, output_size);
    g_free(header);
    g_free(output);
}

/* Meter native samples, never the decimated preview or synthetic ROI exterior. */
static void command_meter(guint request_id, gchar **parts) {
    if (g_strv_length(parts) != 7) { send_error(request_id, "Invalid metering ROI"); return; }
    const gboolean circle = !strcmp(parts[2], "circle");
    if (!circle && strcmp(parts[2], "rectangle")) { send_error(request_id, "Invalid ROI shape"); return; }
    double a[4];
    for (int i = 0; i < 4; i++) {
        gchar *end = NULL;
        a[i] = g_ascii_strtod(parts[i+3], &end);
        if (!end || *end || !isfinite(a[i]) || a[i] < 0 || a[i] > 1) {
            send_error(request_id, "Invalid normalized ROI"); return;
        }
    }
    if ((circle && a[2] <= 0) || (!circle && (a[2] <= a[0] || a[3] <= a[1]))) {
        send_error(request_id, "Empty metering ROI"); return;
    }
    guint64 *histogram = g_new0(guint64, 65536), count = 0, clipped = 0;
    guint peak = 0;
    g_mutex_lock(&bridge.frame_lock);
    const RawFrame *frame = &bridge.latest;
    if (!frame->bytes) {
        g_mutex_unlock(&bridge.frame_lock); g_free(histogram);
        send_error(request_id, "No complete camera frame was received"); return;
    }
    const guint maximum = frame->pixel.sensor_bits <= 8 ? 255 : (65535u << (16-frame->pixel.sensor_bits)) & 65535u;
    /* The delivered samples: one color site in single-channel output. */
    const gint width = output_width(frame), height = output_height(frame);
    const double radius = a[2] * MIN(width, height);
    for (gint y = 0; y < height; y++) {
        for (gint x = 0; x < width; x++) {
            const double dx = x+.5-a[0]*width, dy = y+.5-a[1]*height;
            if (circle ? dx*dx+dy*dy > radius*radius :
                (x+.5 < a[0]*width || x+.5 >= a[2]*width ||
                 y+.5 < a[1]*height || y+.5 >= a[3]*height)) continue;
            const guint value = output_sample(frame, x, y);
            histogram[value]++; count++; peak = MAX(peak, value);
            if (value >= maximum) clipped++;
        }
    }
    const guint64 sequence = frame->sequence;
    g_mutex_unlock(&bridge.frame_lock);
    if (!count) { g_free(histogram); send_error(request_id, "Empty metering ROI"); return; }
    guint percentile = 0; guint64 cumulative = 0;
    for (; percentile < 65535; percentile++) {
        cumulative += histogram[percentile];
        if (cumulative >= (guint64)ceil(count * .999)) break;
    }
    g_free(histogram);
    gchar *result = g_strdup_printf("{\"count\":%" G_GUINT64_FORMAT ",\"clipped\":%" G_GUINT64_FORMAT
        ",\"peak\":%.10g,\"percentile\":%.10g,\"sequence\":%" G_GUINT64_FORMAT "}",
        count, clipped, (double)peak/maximum, (double)percentile/maximum, sequence);
    send_result(request_id, result); g_free(result);
}

static uint16_t focus_sample(const void *context, int x, int y) {
    return output_sample((const RawFrame *)context, x, y);
}

/* Command-driven, bounded work. The renderer never receives native image data
 * just to calculate focus, and no synthetic ROI border enters the metric. */
static void command_focus(guint request_id, gchar **parts) {
    if (g_strv_length(parts) != 7) { send_error(request_id, "Invalid focus ROI"); return; }
    FocusRoi roi = {0};
    roi.circle = !strcmp(parts[2], "circle");
    if (!roi.circle && strcmp(parts[2], "rectangle")) { send_error(request_id, "Invalid ROI shape"); return; }
    double a[4];
    for (int i = 0; i < 4; i++) {
        gchar *end = NULL;
        a[i] = g_ascii_strtod(parts[i+3], &end);
        if (!end || *end || !isfinite(a[i]) || a[i] < 0 || a[i] > 1) {
            send_error(request_id, "Invalid normalized ROI"); return;
        }
    }
    if ((roi.circle && a[2] <= 0) || (!roi.circle && (a[2] <= a[0] || a[3] <= a[1]))) {
        send_error(request_id, "Empty focus ROI"); return;
    }
    roi.a = a[0]; roi.b = a[1]; roi.c = a[2]; roi.d = a[3];
    g_mutex_lock(&bridge.frame_lock);
    const RawFrame *frame = &bridge.latest;
    if (!frame->bytes) {
        g_mutex_unlock(&bridge.frame_lock);
        send_error(request_id, "No complete camera frame was received"); return;
    }
    const guint maximum = frame->pixel.sensor_bits <= 8 ? 255 : (65535u << (16-frame->pixel.sensor_bits)) & 65535u;
    const FocusResult focus = measure_focus(output_width(frame), output_height(frame), maximum, roi, focus_sample, frame);
    const guint64 sequence = frame->sequence;
    g_mutex_unlock(&bridge.frame_lock);
    gchar *result = g_strdup_printf("{\"score\":%.10g,\"count\":%u,\"status\":\"%s\",\"sequence\":%" G_GUINT64_FORMAT "}",
                                    focus.score, focus.count, focus.status, sequence);
    send_result(request_id, result); g_free(result);
}

static gboolean parse_bool(const gchar *value) {
    return !g_ascii_strcasecmp(value, "true") || !strcmp(value, "1") || !g_ascii_strcasecmp(value, "on");
}

static void command_set(guint request_id, const gchar *encoded_name, const gchar *encoded_value) {
    if (!bridge.camera) { send_error(request_id, "Open a camera before changing controls"); return; }
    gchar *name = g_uri_unescape_string(encoded_name, NULL), *value = g_uri_unescape_string(encoded_value, NULL);
    if (!name || !value) { send_error(request_id, "Invalid control command"); g_free(name); g_free(value); return; }
    GError *error = NULL;
    if (!strcmp(name, "channel")) {
        gint channel = -1;
        for (gint i = 0; i < (gint)G_N_ELEMENTS(channel_names); i++) if (!strcmp(value, channel_names[i])) channel = i;
        if (channel < 0) g_set_error(&error, G_IO_ERROR, G_IO_ERROR_INVALID_ARGUMENT, "Unknown camera channel: %s", value);
        else configure_output((OutputChannel)channel, &error);
        if (error) send_gerror(request_id, "Unable to select camera channel", error);
        else {
            gchar *camera = camera_json(), *controls = controls_json();
            gchar *result = g_strdup_printf("{\"camera\":%s,\"controls\":%s}", camera, controls);
            send_result(request_id, result);
            g_free(result); g_free(camera); g_free(controls);
        }
        g_free(name); g_free(value);
        return;
    }
    if (!strcmp(name, "exposureAuto")) arv_camera_set_exposure_time_auto(bridge.camera, parse_bool(value) ? ARV_AUTO_CONTINUOUS : ARV_AUTO_OFF, &error);
    else if (!strcmp(name, "gainAuto")) arv_camera_set_gain_auto(bridge.camera, parse_bool(value) ? ARV_AUTO_CONTINUOUS : ARV_AUTO_OFF, &error);
    else if (!strcmp(name, "exposure")) arv_camera_set_exposure_time(bridge.camera, g_ascii_strtod(value, NULL), &error);
    else if (!strcmp(name, "gain")) arv_camera_set_gain(bridge.camera, g_ascii_strtod(value, NULL), &error);
    else g_set_error(&error, G_IO_ERROR, G_IO_ERROR_INVALID_ARGUMENT, "Unknown camera control: %s", name);
    if (error) send_gerror(request_id, "Unable to set camera control", error);
    else {
        gchar *controls = controls_json();
        gchar *result = g_strdup_printf("{\"controls\":%s}", controls);
        send_result(request_id, result);
        g_free(result); g_free(controls);
    }
    g_free(name); g_free(value);
}

int main(void) {
#ifdef G_OS_WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    memset(&bridge, 0, sizeof(bridge));
    bridge.step = 1;
    g_mutex_init(&bridge.output_lock);
    g_mutex_init(&bridge.frame_lock);
    if (g_getenv("INSPECTION_CAMERA_ENABLE_FAKE")) arv_enable_interface("Fake");
    setvbuf(stdout, NULL, _IONBF, 0);
    gchar line[8192];
    gboolean running = TRUE;
    while (running && fgets(line, sizeof(line), stdin)) {
        g_strchomp(line);
        gchar **parts = g_strsplit(line, "\t", 0);
        const guint request_id = parts[1] ? (guint)g_ascii_strtoull(parts[1], NULL, 10) : 0;
        if (!parts[0] || !request_id) send_error(request_id, "Malformed camera bridge command");
        else if (!strcmp(parts[0], "discover")) command_discover(request_id);
        else if (!strcmp(parts[0], "open") && parts[2]) command_open(request_id, parts[2]);
        else if (!strcmp(parts[0], "start")) command_start(request_id);
        else if (!strcmp(parts[0], "meter")) command_meter(request_id, parts);
        else if (!strcmp(parts[0], "focus")) command_focus(request_id, parts);
        else if (!strcmp(parts[0], "pause")) command_pause(request_id);
        else if (!strcmp(parts[0], "set") && parts[2] && parts[3]) command_set(request_id, parts[2], parts[3]);
        else if (!strcmp(parts[0], "quit")) { send_result(request_id, "{}"); running = FALSE; }
        else send_error(request_id, "Unknown or incomplete camera bridge command");
        g_strfreev(parts);
    }
    close_camera();
    arv_shutdown();
    g_mutex_clear(&bridge.frame_lock);
    g_mutex_clear(&bridge.output_lock);
    return 0;
}
