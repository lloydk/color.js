import { to, sRGB, HPLuv } from "../../src/index-fn.js";
import * as check from "../../node_modules/htest.dev/src/check.js";
import { readTestData } from "./hsluv.js";

let json = readTestData();
let srgbToHpluv = [];
let hpluvToSrgb = [];

Object.entries(json).forEach(([rgbHex, value]) => {
	let coords = value.hpluv;
	let rgb = value.rgb;
	srgbToHpluv.push({ args: { space: sRGB, coords: rgb }, expect: coords });
	hpluvToSrgb.push({ args: { space: HPLuv, coords: coords }, expect: rgb });
});

const tests = {
	name: "HPLuv Conversion Tests",
	description:
		"These tests compare sRGB values against the HSLuv reference implementation snapshot data.",
	run (color, space = this.data.toSpace) {
		return space.from(color.space, color.coords);
	},
	check: check.deep(function (actual, expect) {
		if (expect === null || Number.isNaN(expect)) {
			// Treat NaN and null as equivalent for now
			return actual === null || Number.isNaN(actual);
		}

		let checkProximity = check.proximity({ epsilon: this.data.epsilon });
		return checkProximity(actual, expect);
	}),
	data: {
		epsilon: 0.00000001,
	},
	tests: [
		{
			name: "sRGB to HPLuv",
			data: {
				toSpace: HPLuv,
			},
			tests: srgbToHpluv,
		},
		{
			name: "HPLuv to sRGB",
			data: {
				toSpace: sRGB,
			},
			tests: hpluvToSrgb,
		},
	],
};

export default tests;
