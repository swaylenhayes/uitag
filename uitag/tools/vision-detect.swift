#!/usr/bin/env swift
// SPDX-License-Identifier: MIT
// Apple Vision text + rectangle detection tool.
// Outputs JSON detections to stdout, timing to stderr.

import AppKit
import Foundation
import Vision

// MARK: - Helpers

struct DetectionItem: Codable {
    let label: String
    let x: Int
    let y: Int
    let width: Int
    let height: Int
    let confidence: Double
    let source: String
}

struct Output: Codable {
    let image_width: Int
    let image_height: Int
    let detections: [DetectionItem]
}

func visionRectToPixel(
    _ rect: CGRect, imageWidth: Int, imageHeight: Int
) -> (x: Int, y: Int, width: Int, height: Int) {
    // Vision coordinates: origin bottom-left, normalized 0-1.
    // Convert to pixel coordinates with origin top-left.
    let pixelX = Int(rect.origin.x * Double(imageWidth))
    let pixelY = Int((1.0 - rect.origin.y - rect.height) * Double(imageHeight))
    let pixelW = Int(rect.size.width * Double(imageWidth))
    let pixelH = Int(rect.size.height * Double(imageHeight))
    return (pixelX, pixelY, pixelW, pixelH)
}

// MARK: - Main

guard CommandLine.arguments.count >= 2 else {
    fputs("Usage: vision-detect <image-path> [--fast]\n", stderr)
    exit(1)
}

let imagePath = CommandLine.arguments[1]
let useFastMode = CommandLine.arguments.contains("--fast")

guard let nsImage = NSImage(contentsOfFile: imagePath) else {
    fputs("Error: could not load image at \(imagePath)\n", stderr)
    exit(1)
}

guard let tiffData = nsImage.tiffRepresentation,
      let bitmapRep = NSBitmapImageRep(data: tiffData),
      let cgImage = bitmapRep.cgImage(forProposedRect: nil, context: nil, hints: nil)
else {
    fputs("Error: could not convert image to CGImage\n", stderr)
    exit(1)
}

let imageWidth = cgImage.width
let imageHeight = cgImage.height

let startTime = CFAbsoluteTimeGetCurrent()

// MARK: - Text Recognition

let textRequest = VNRecognizeTextRequest()
textRequest.recognitionLevel = useFastMode ? .fast : .accurate
textRequest.usesLanguageCorrection = !useFastMode

// MARK: - Rectangle Detection

let rectRequest = VNDetectRectanglesRequest()
rectRequest.minimumSize = 0.02
rectRequest.maximumObservations = 50
rectRequest.minimumConfidence = 0.5
rectRequest.minimumAspectRatio = 0.1
rectRequest.maximumAspectRatio = 10.0

// MARK: - Execute

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

do {
    try handler.perform([textRequest, rectRequest])
} catch {
    fputs("Error: Vision request failed — \(error.localizedDescription)\n", stderr)
    exit(1)
}

var detections: [DetectionItem] = []

// Process text results
let textResults = textRequest.results ?? []
for observation in textResults {
    let text = observation.topCandidates(1).first?.string ?? ""
    let bbox = observation.boundingBox
    let conf = Double(observation.confidence)
    let (px, py, pw, ph) = visionRectToPixel(bbox, imageWidth: imageWidth, imageHeight: imageHeight)
    detections.append(DetectionItem(
        label: text, x: px, y: py, width: pw, height: ph,
        confidence: conf, source: "vision_text"
    ))
}

// Process rectangle results
let rectResults = rectRequest.results ?? []
for observation in rectResults {
    let bbox = observation.boundingBox
    let conf = Double(observation.confidence)
    let (px, py, pw, ph) = visionRectToPixel(bbox, imageWidth: imageWidth, imageHeight: imageHeight)
    detections.append(DetectionItem(
        label: "rectangle", x: px, y: py, width: pw, height: ph,
        confidence: conf, source: "vision_rect"
    ))
}

let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

// MARK: - Output

let output = Output(image_width: imageWidth, image_height: imageHeight, detections: detections)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

guard let jsonData = try? encoder.encode(output),
      let jsonString = String(data: jsonData, encoding: .utf8)
else {
    fputs("Error: failed to encode JSON\n", stderr)
    exit(1)
}

print(jsonString)

// Timing to stderr
fputs("vision_time_ms=\(String(format: "%.1f", elapsed))\n", stderr)
fputs("text_count=\(textResults.count)\n", stderr)
fputs("rect_count=\(rectResults.count)\n", stderr)
